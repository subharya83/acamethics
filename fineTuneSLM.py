import argparse
import json
import os
import logging
import platform
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

# Use the modern ``evaluate`` library instead of the deprecated
# ``datasets.load_metric``.
import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MathQADataset(Dataset):
    """Custom Dataset for question answering on math problems."""

    def __init__(self, qa_pairs, tokenizer, max_length=512):
        """Initialize dataset with QA pairs and tokenizer.

        Args:
            qa_pairs (list): List of dicts with question, answer, and context.
            tokenizer: Transformers tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        self.start_positions = []
        self.end_positions = []

        skipped = 0
        for qa_pair in tqdm(qa_pairs, desc="Tokenizing dataset"):
            question = qa_pair["question"]
            context = qa_pair["context"]
            answer = qa_pair["answer"]

            encoding = tokenizer(
                question,
                context,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            start_pos, end_pos = self._find_answer_positions(
                context, answer, encoding
            )

            # Skip samples where the answer could not be located.
            # The original code silently inserted (0, 0) which points at
            # [CLS] and corrupts training.
            if start_pos == 0 and end_pos == 0 and answer.strip():
                skipped += 1
                continue

            self.encodings.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                }
            )
            self.start_positions.append(start_pos)
            self.end_positions.append(end_pos)

        if skipped > 0:
            logger.warning(
                "Skipped %d samples where the answer could not be "
                "mapped to token positions",
                skipped,
            )

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.encodings[idx].items()}
        item["start_positions"] = self.start_positions[idx]
        item["end_positions"] = self.end_positions[idx]
        return item

    @staticmethod
    def _find_answer_positions(context, answer, encoding):
        """Find token positions of the answer in the context.

        Returns:
            tuple: (start_token, end_token) indices, or (0, 0) on failure.
        """
        offset_mapping = encoding["offset_mapping"][0]

        start_char = context.find(answer)
        if start_char == -1:
            logger.warning("Answer '%s' not found in context", answer)
            return 0, 0

        end_char = start_char + len(answer) - 1

        start_token = None
        end_token = None
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                # Special token or padding
                continue
            # Use <= on right boundary to avoid off-by-one when the
            # answer's last character aligns exactly with a token edge.
            if start_token is None and tok_start <= start_char < tok_end:
                start_token = idx
            if tok_start <= end_char < tok_end:
                end_token = idx

        if start_token is None or end_token is None:
            logger.warning("Could not map answer '%s' to tokens", answer)
            return 0, 0

        return start_token, end_token


def load_qa_pairs(input_dir):
    """Load QA pairs from JSON files in the input directory.

    Args:
        input_dir (str): Directory containing JSON files with QA pairs.

    Returns:
        list: List of QA pairs.
    """
    qa_pairs = []
    for filename in tqdm(os.listdir(input_dir), desc="Loading JSON files"):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for pair in data.get("qa_pairs", []):
                        if all(
                            key in pair
                            for key in ["question", "answer", "context"]
                        ):
                            qa_pairs.append(pair)
                        else:
                            logger.warning(
                                "Skipping invalid QA pair in %s: %s",
                                filename,
                                pair,
                            )
            except json.JSONDecodeError as e:
                logger.error("Error reading %s: %s", filename, e)
    return qa_pairs


def build_compute_metrics():
    """Return a compute_metrics function that compares predicted vs true
    span positions using the SQuAD metric.

    The original implementation created empty placeholder predictions so
    F1 and EM were always 0.  This version encodes span positions as
    text so the SQuAD string-match metric produces meaningful scores.
    """
    squad_metric = evaluate.load("squad")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        start_preds, end_preds = predictions
        start_labels, end_labels = labels

        pred_starts = np.argmax(start_preds, axis=1)
        pred_ends = np.argmax(end_preds, axis=1)

        formatted_preds = []
        formatted_refs = []
        for i in range(len(pred_starts)):
            pred_text = f"{int(pred_starts[i])}-{int(pred_ends[i])}"
            ref_text = f"{int(start_labels[i])}-{int(end_labels[i])}"
            formatted_preds.append(
                {"prediction_text": pred_text, "id": str(i)}
            )
            formatted_refs.append(
                {
                    "answers": {
                        "text": [ref_text],
                        "answer_start": [int(start_labels[i])],
                    },
                    "id": str(i),
                }
            )

        return squad_metric.compute(
            predictions=formatted_preds, references=formatted_refs
        )

    return compute_metrics


def predict(model, tokenizer, question, context, device, max_length=512):
    """Perform inference on a question-context pair.

    Args:
        model: Trained QA model.
        tokenizer: Transformers tokenizer instance.
        question (str): Question text.
        context (str): Context text.
        device: PyTorch device.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        str: Predicted answer.
    """
    model.eval()
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        start_pos = torch.argmax(outputs.start_logits, dim=1).item()
        end_pos = torch.argmax(outputs.end_logits, dim=1).item()

    # Clamp: end must be >= start
    if end_pos < start_pos:
        end_pos = start_pos

    answer_tokens = inputs["input_ids"][0][start_pos : end_pos + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer


def main():
    """Fine-tune a DistilBERT model for math question answering."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for math QA"
    )
    parser.add_argument(
        "-i", "--input-dir", required=True,
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "-t", "--tmp-dir", required=True,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "-w", "--output-dir", required=True,
        help="Directory to save pre-trained and final model",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Path to pre-trained model checkpoint",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Number of gradient accumulation steps (useful for limited VRAM)",
    )
    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist")
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    if args.resume_from:
        logger.info("Loading model and tokenizer from %s", args.resume_from)
        tokenizer = DistilBertTokenizer.from_pretrained(args.resume_from)
        model = DistilBertForQuestionAnswering.from_pretrained(
            args.resume_from
        ).to(device)
    else:
        logger.info("Downloading tokenizer and model to %s", args.output_dir)
        tokenizer = DistilBertTokenizer.from_pretrained(
            model_name, cache_dir=args.output_dir
        )
        model = DistilBertForQuestionAnswering.from_pretrained(
            model_name, cache_dir=args.output_dir
        ).to(device)

    # Load and split dataset
    qa_pairs = load_qa_pairs(args.input_dir)
    logger.info("Loaded %d QA pairs", len(qa_pairs))

    if len(qa_pairs) < 5:
        raise ValueError(
            "Too few QA pairs for training. Need at least 5, got %d."
            % len(qa_pairs)
        )

    train_pairs, val_pairs = train_test_split(
        qa_pairs, test_size=0.2, random_state=42
    )
    logger.info(
        "Training set: %d pairs, Validation set: %d pairs",
        len(train_pairs),
        len(val_pairs),
    )
    train_dataset = MathQADataset(train_pairs, tokenizer)
    val_dataset = MathQADataset(val_pairs, tokenizer)

    if len(train_dataset) == 0:
        raise ValueError(
            "No valid training samples after tokenization. "
            "Check that answers appear verbatim in their contexts."
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.tmp_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.tmp_dir, "logs"),
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4 if platform.system() != "Darwin" else 0,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=build_compute_metrics(),
    )

    # Train and evaluate
    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)

    # Save model
    logger.info("Saving fine-tuned model to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Test inference
    logger.info("Testing inference...")
    question = "What is 2 + 2?"
    context = "2 + 2 equals 4."
    answer = predict(model, tokenizer, question, context, device)
    logger.info("Question: %s\nAnswer: %s", question, answer)


if __name__ == "__main__":
    main()