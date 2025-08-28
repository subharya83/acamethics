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
from datasets import load_metric

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MathQADataset(Dataset):
    """Custom Dataset for question answering on math problems."""
    
    def __init__(self, qa_pairs, tokenizer, max_length=512):
        """Initialize dataset with QA pairs and tokenizer.
        
        Args:
            qa_pairs (list): List of dictionaries containing question, answer, and context.
            tokenizer: Transformers tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        self.start_positions = []
        self.end_positions = []

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

            start_pos, end_pos = self.find_answer_positions(context, answer, tokenizer, encoding)
            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
            })
            self.start_positions.append(start_pos)
            self.end_positions.append(end_pos)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.encodings)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        encoding = self.encodings[idx]
        encoding["start_positions"] = self.start_positions[idx]
        encoding["end_positions"] = self.end_positions[idx]
        return encoding

    def find_answer_positions(self, context, answer, tokenizer, encoding):
        """Find token positions of the answer in the context.
        
        Args:
            context (str): Context text.
            answer (str): Answer text.
            tokenizer: Transformers tokenizer instance.
            encoding: Tokenized encoding from the tokenizer.
        
        Returns:
            tuple: Start and end token positions of the answer.
        """
        context_chars = context
        answer_chars = answer
        offset_mapping = encoding["offset_mapping"][0]

        start_char = context_chars.find(answer_chars)
        if start_char == -1:
            logger.warning(f"Answer '{answer}' not found in context")
            return 0, 0

        end_char = start_char + len(answer_chars) - 1

        # Map character positions to token positions
        start_token = None
        end_token = None
        for idx, (start, end) in enumerate(offset_mapping):
            if start_token is None and start <= start_char < end:
                start_token = idx
            if end_token is None and start <= end_char < end:
                end_token = idx
                break

        if start_token is None or end_token is None:
            logger.warning(f"Could not map answer '{answer}' to tokens")
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
                        if all(key in pair for key in ["question", "answer", "context"]):
                            qa_pairs.append(pair)
                        else:
                            logger.warning(f"Skipping invalid QA pair in {filename}: {pair}")
            except json.JSONDecodeError as e:
                logger.error(f"Error reading {filename}: {e}")
    return qa_pairs

def compute_metrics(eval_pred):
    """Compute evaluation metrics for QA task.
    
    Args:
        eval_pred: Tuple of predictions and labels from the Trainer.
    
    Returns:
        dict: Metrics including F1 and Exact Match.
    """
    metric = load_metric("squad")
    predictions, labels = eval_pred
    start_preds, end_preds = predictions
    start_labels, end_labels = labels

    # Simplified: Actual implementation should map predictions back to text
    predictions = [{"prediction_text": "", "id": str(i)} for i in range(len(start_preds))]
    references = [{"answers": {"text": [""], "answer_start": [0]}, "id": str(i)} for i in range(len(start_labels))]
    return metric.compute(predictions=predictions, references=references)

def predict(model, tokenizer, question, context, device, max_length=512):
    """Perform inference on a question-context pair.
    
    Args:
        model: Trained QA model.
        tokenizer: Transformers tokenizer instance.
        question (str): Question text.
        context (str): Context text.
        device: PyTorch device (e.g., cuda, mps, cpu).
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
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_pos = torch.argmax(start_logits, dim=1).item()
        end_pos = torch.argmax(end_logits, dim=1).item()

    answer_tokens = inputs["input_ids"][0][start_pos:end_pos + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

def main():
    """Fine-tune a DistilBERT model for 6th-grade math question answering."""
    parser = argparse.ArgumentParser(description="Fine-tune an SLM for 6th-grade math QA")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing JSON files")
    parser.add_argument("-t", "--tmp-dir", required=True, help="Directory for saving checkpoints")
    parser.add_argument("-w", "--output-dir", required=True, help="Directory to save pre-trained and final model")
    parser.add_argument("--resume-from", default=None, help="Path to pre-trained model checkpoint")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist")
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    if args.resume_from:
        logger.info(f"Loading model and tokenizer from {args.resume_from}")
        tokenizer = DistilBertTokenizer.from_pretrained(args.resume_from)
        model = DistilBertForQuestionAnswering.from_pretrained(args.resume_from).to(device)
    else:
        logger.info(f"Downloading tokenizer and model to {args.output_dir}")
        tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=args.output_dir)
        model = DistilBertForQuestionAnswering.from_pretrained(model_name, cache_dir=args.output_dir).to(device)

    # Load and split dataset
    qa_pairs = load_qa_pairs(args.input_dir)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    train_pairs, val_pairs = train_test_split(qa_pairs, test_size=0.2, random_state=42)
    logger.info(f"Training set: {len(train_pairs)} pairs, Validation set: {len(val_pairs)} pairs")
    train_dataset = MathQADataset(train_pairs, tokenizer)
    val_dataset = MathQADataset(val_pairs, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.tmp_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.tmp_dir, "logs"),
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="steps",
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
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    # Save model
    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Test inference
    logger.info("Testing inference...")
    question = "What is 2 + 2?"
    context = "2 + 2 equals 4."
    answer = predict(model, tokenizer, question, context, device)
    logger.info(f"Question: {question}\nAnswer: {answer}")

if __name__ == "__main__":
    main()