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
import evaluate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"question", "answer", "context"}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_qa_schema(qa_pairs: list) -> list:
    """Validate and filter QA pairs for schema correctness.

    Checks:
      - Required keys present (question, answer, context)
      - Non-empty strings for each key
      - Answer is a verbatim substring of context (extractive compatibility)

    Returns only the pairs that pass all checks, logging warnings for failures.
    """
    valid = []
    schema_fail = 0
    empty_fail = 0
    substr_fail = 0

    for i, pair in enumerate(qa_pairs):
        if not REQUIRED_KEYS.issubset(pair.keys()):
            schema_fail += 1
            continue
        q = pair["question"].strip()
        a = pair["answer"].strip()
        c = pair["context"].strip()
        if not q or not a or not c:
            empty_fail += 1
            continue
        if a not in c:
            substr_fail += 1
            continue
        valid.append({"question": q, "answer": a, "context": c})

    total_dropped = schema_fail + empty_fail + substr_fail
    if total_dropped:
        logger.warning(
            "Schema validation dropped %d pairs: %d missing keys, "
            "%d empty fields, %d answer-not-in-context",
            total_dropped, schema_fail, empty_fail, substr_fail,
        )
    logger.info("Schema validation passed: %d / %d pairs", len(valid), len(qa_pairs))
    return valid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MathQADataset(Dataset):
    """Custom Dataset for extractive QA on math problems."""

    def __init__(self, qa_pairs, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        self.start_positions = []
        self.end_positions = []
        # Keep raw pairs for post-training evaluation
        self.raw_pairs = []

        skipped = 0
        for qa_pair in tqdm(qa_pairs, desc="Tokenizing dataset"):
            question = qa_pair["question"]
            context = qa_pair["context"]
            answer = qa_pair["answer"]

            encoding = tokenizer(
                question, context,
                max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
                return_offsets_mapping=True,
            )

            start_pos, end_pos = self._find_answer_positions(context, answer, encoding)

            if start_pos == 0 and end_pos == 0 and answer.strip():
                skipped += 1
                continue

            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
            })
            self.start_positions.append(start_pos)
            self.end_positions.append(end_pos)
            self.raw_pairs.append(qa_pair)

        if skipped:
            logger.warning("Skipped %d samples (answer not mappable to tokens)", skipped)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.encodings[idx].items()}
        item["start_positions"] = self.start_positions[idx]
        item["end_positions"] = self.end_positions[idx]
        return item

    @staticmethod
    def _find_answer_positions(context, answer, encoding):
        offset_mapping = encoding["offset_mapping"][0]
        start_char = context.find(answer)
        if start_char == -1:
            return 0, 0
        end_char = start_char + len(answer) - 1

        start_token = None
        end_token = None
        for idx, (ts, te) in enumerate(offset_mapping):
            if ts == te:
                continue
            if start_token is None and ts <= start_char < te:
                start_token = idx
            if ts <= end_char < te:
                end_token = idx

        if start_token is None or end_token is None:
            return 0, 0
        return start_token, end_token


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_qa_pairs(input_dir: str) -> list:
    """Load QA pairs from JSON files in the input directory."""
    qa_pairs = []
    for filename in tqdm(os.listdir(input_dir), desc="Loading JSON files"):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(input_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for pair in data.get("qa_pairs", []):
                    qa_pairs.append(pair)
        except json.JSONDecodeError as e:
            logger.error("Error reading %s: %s", filename, e)
    return qa_pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics():
    """Position-level SQuAD metric for Trainer callbacks."""
    squad_metric = evaluate.load("squad")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        start_preds, end_preds = predictions
        start_labels, end_labels = labels
        ps = np.argmax(start_preds, axis=1)
        pe = np.argmax(end_preds, axis=1)
        preds = [{"prediction_text": f"{int(ps[i])}-{int(pe[i])}", "id": str(i)} for i in range(len(ps))]
        refs = [{"answers": {"text": [f"{int(start_labels[i])}-{int(end_labels[i])}"],
                              "answer_start": [int(start_labels[i])]}, "id": str(i)} for i in range(len(ps))]
        return squad_metric.compute(predictions=preds, references=refs)

    return compute_metrics


# ---------------------------------------------------------------------------
# Text-level post-training evaluation
# ---------------------------------------------------------------------------

def compute_text_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between predicted and gold answer strings."""
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_text_level(model, tokenizer, dataset, device, max_length=512):
    """Run inference on every sample in dataset and compute text F1 and EM.

    This gives a human-interpretable evaluation unlike the position-level
    metric used during training.
    """
    model.eval()
    f1_scores = []
    exact_matches = 0
    total = len(dataset.raw_pairs)

    for i in tqdm(range(total), desc="Text-level evaluation"):
        qa = dataset.raw_pairs[i]
        inputs = tokenizer(
            qa["question"], qa["context"],
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            s = torch.argmax(outputs.start_logits, dim=1).item()
            e = torch.argmax(outputs.end_logits, dim=1).item()
        if e < s:
            e = s
        tokens = inputs["input_ids"][0][s : e + 1]
        pred = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        gold = qa["answer"].strip()

        f1 = compute_text_f1(pred, gold)
        f1_scores.append(f1)
        if pred.lower() == gold.lower():
            exact_matches += 1

    avg_f1 = sum(f1_scores) / max(len(f1_scores), 1)
    em = exact_matches / max(total, 1)
    return {"text_f1": avg_f1, "text_em": em, "total_samples": total}


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(model, tokenizer, question, context, device, max_length=512):
    model.eval()
    inputs = tokenizer(
        question, context,
        max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        s = torch.argmax(outputs.start_logits, dim=1).item()
        e = torch.argmax(outputs.end_logits, dim=1).item()
    if e < s:
        e = s
    tokens = inputs["input_ids"][0][s : e + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for math QA")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing JSON files")
    parser.add_argument("-t", "--tmp-dir", required=True, help="Directory for checkpoints")
    parser.add_argument("-w", "--output-dir", required=True, help="Directory for final model")
    parser.add_argument("--resume-from", default=None, help="Checkpoint to resume from")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist")
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Load and validate
    raw_pairs = load_qa_pairs(args.input_dir)
    logger.info("Loaded %d raw QA pairs", len(raw_pairs))
    qa_pairs = validate_qa_schema(raw_pairs)

    if len(qa_pairs) < 5:
        raise ValueError("Too few valid QA pairs for training (%d). Need >= 5." % len(qa_pairs))

    # Model
    model_name = "distilbert-base-uncased"
    if args.resume_from:
        logger.info("Resuming from %s", args.resume_from)
        tokenizer = DistilBertTokenizer.from_pretrained(args.resume_from)
        model = DistilBertForQuestionAnswering.from_pretrained(args.resume_from).to(device)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=args.output_dir)
        model = DistilBertForQuestionAnswering.from_pretrained(model_name, cache_dir=args.output_dir).to(device)

    # Split
    train_pairs, val_pairs = train_test_split(qa_pairs, test_size=0.2, random_state=42)
    logger.info("Train: %d, Val: %d", len(train_pairs), len(val_pairs))
    train_dataset = MathQADataset(train_pairs, tokenizer)
    val_dataset = MathQADataset(val_pairs, tokenizer)

    if len(train_dataset) == 0:
        raise ValueError("No valid training samples after tokenization.")

    # Train
    training_args = TrainingArguments(
        output_dir=args.tmp_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=500, weight_decay=0.01,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.tmp_dir, "logs"),
        logging_steps=100, save_steps=1000, save_total_limit=2,
        eval_strategy="steps", eval_steps=500,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4 if platform.system() != "Darwin" else 0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=build_compute_metrics(),
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    logger.info("Trainer evaluation (position-level)...")
    eval_results = trainer.evaluate()
    logger.info("Eval results: %s", eval_results)

    # Save
    logger.info("Saving model to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Post-training text-level evaluation on validation set
    logger.info("Running text-level evaluation on validation set...")
    text_metrics = evaluate_text_level(model, tokenizer, val_dataset, device)
    logger.info(
        "Text-level results: F1=%.4f, EM=%.4f (%d samples)",
        text_metrics["text_f1"], text_metrics["text_em"], text_metrics["total_samples"],
    )

    # Save evaluation report
    report_path = os.path.join(args.output_dir, "eval_report.json")
    report = {
        "trainer_eval": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in eval_results.items()},
        "text_level_eval": text_metrics,
        "dataset_stats": {
            "total_loaded": len(raw_pairs),
            "after_schema_validation": len(qa_pairs),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report saved to %s", report_path)

    # Quick inference test
    logger.info("Quick inference test...")
    answer = predict(model, tokenizer, "What is 2 + 2?", "2 + 2 equals 4.", device)
    logger.info("Q: What is 2 + 2? -> A: %s", answer)


if __name__ == "__main__":
    main()