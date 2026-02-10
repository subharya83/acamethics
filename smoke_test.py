#!/usr/bin/env python3
"""End-to-end smoke test for the Acamethics pipeline.

Runs a minimal version of the full workflow using sample_qa.json:
  1. Schema validation
  2. Fine-tuning for 1 epoch on the sample data
  3. Text-level evaluation on the validation split
  4. Single-question inference
  5. Chunk retrieval (TF-IDF) test

Usage:
    python smoke_test.py [--sample-data path/to/sample_qa.json]

This test is designed to run on CPU in under 5 minutes.
"""

import argparse
import json
import os
import sys
import shutil
import tempfile
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PASS_COUNT = 0
FAIL_COUNT = 0


def record(passed: bool):
    global PASS_COUNT, FAIL_COUNT
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1


def check_imports() -> bool:
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "evaluate": "evaluate",
        "sklearn": "scikit-learn",
        "tqdm": "tqdm",
        "numpy": "numpy",
        "flask": "flask",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing packages: %s", ", ".join(missing))
        logger.error("Install: pip install %s", " ".join(missing))
        return False
    return True


# ------------------------------------------------------------------
# Test 1
# ------------------------------------------------------------------
def test_schema_validation(sample_path: str) -> bool:
    logger.info("=" * 50)
    logger.info("TEST 1: Schema Validation")
    logger.info("=" * 50)

    with open(sample_path, "r") as f:
        data = json.load(f)

    pairs = data.get("qa_pairs", [])
    if not pairs:
        logger.error("FAIL: No qa_pairs found")
        return False

    for i, pair in enumerate(pairs):
        for key in ("question", "answer", "context"):
            if key not in pair:
                logger.error("FAIL: pair %d missing key '%s'", i, key)
                return False
        a = pair["answer"].strip()
        if a not in pair["context"]:
            logger.error("FAIL: pair %d answer not in context", i)
            return False

    logger.info("PASS: %d/%d pairs validated", len(pairs), len(pairs))
    return True


# ------------------------------------------------------------------
# Test 2
# ------------------------------------------------------------------
def test_fine_tuning(sample_path: str, tmp_dir: str) -> str:
    """Returns model output directory or empty string on failure."""
    logger.info("=" * 50)
    logger.info("TEST 2: Fine-tuning (1 epoch, CPU)")
    logger.info("=" * 50)

    from fineTuneSLM import (
        load_qa_pairs, validate_qa_schema, MathQADataset,
    )
    from transformers import (
        DistilBertTokenizer, DistilBertForQuestionAnswering,
        Trainer, TrainingArguments,
    )
    import torch

    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(sample_path, os.path.join(data_dir, "sample_qa.json"))

    raw = load_qa_pairs(data_dir)
    pairs = validate_qa_schema(raw)

    if len(pairs) < 5:
        logger.error("FAIL: Not enough valid pairs (%d)", len(pairs))
        return ""

    from sklearn.model_selection import train_test_split
    model_name = "distilbert-base-uncased"
    output_dir = os.path.join(tmp_dir, "model")
    ckpt_dir = os.path.join(tmp_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=tmp_dir)
    model = DistilBertForQuestionAnswering.from_pretrained(model_name, cache_dir=tmp_dir)
    device = torch.device("cpu")
    model.to(device)

    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    train_ds = MathQADataset(train_pairs, tokenizer)
    val_ds = MathQADataset(val_pairs, tokenizer)

    if len(train_ds) == 0:
        logger.error("FAIL: No training samples after tokenization")
        return ""

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("PASS: Model saved to %s", output_dir)
    return output_dir


# ------------------------------------------------------------------
# Test 3
# ------------------------------------------------------------------
def test_text_evaluation(model_dir: str, sample_path: str) -> bool:
    logger.info("=" * 50)
    logger.info("TEST 3: Text-level Evaluation")
    logger.info("=" * 50)

    from fineTuneSLM import validate_qa_schema, MathQADataset, evaluate_text_level
    from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
    from sklearn.model_selection import train_test_split
    import torch

    with open(sample_path, "r") as f:
        data = json.load(f)
    pairs = validate_qa_schema(data.get("qa_pairs", []))
    _, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForQuestionAnswering.from_pretrained(model_dir)
    device = torch.device("cpu")
    model.to(device)

    val_ds = MathQADataset(val_pairs, tokenizer)
    metrics = evaluate_text_level(model, tokenizer, val_ds, device)

    logger.info(
        "Text F1: %.4f, Text EM: %.4f (%d samples)",
        metrics["text_f1"], metrics["text_em"], metrics["total_samples"],
    )
    if metrics["total_samples"] == 0:
        logger.error("FAIL: No samples evaluated")
        return False

    logger.info("PASS: Text-level evaluation completed")
    return True


# ------------------------------------------------------------------
# Test 4
# ------------------------------------------------------------------
def test_inference(model_dir: str) -> bool:
    logger.info("=" * 50)
    logger.info("TEST 4: Inference")
    logger.info("=" * 50)

    from fineTuneSLM import predict
    from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
    import torch

    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForQuestionAnswering.from_pretrained(model_dir)
    device = torch.device("cpu")
    model.to(device)

    q = "What is a prime number?"
    c = ("A prime number is a natural number greater than 1 that has no "
         "positive divisors other than 1 and itself.")
    answer = predict(model, tokenizer, q, c, device)

    if not answer or not answer.strip():
        logger.error("FAIL: Empty answer returned")
        return False

    logger.info("Q: %s", q)
    logger.info("A: %s", answer)
    logger.info("PASS: Inference produced a non-empty answer")
    return True


# ------------------------------------------------------------------
# Test 5
# ------------------------------------------------------------------
def test_chunk_retrieval(sample_path: str, tmp_dir: str) -> bool:
    logger.info("=" * 50)
    logger.info("TEST 5: Chunk Retrieval (TF-IDF)")
    logger.info("=" * 50)

    # Build a chunk index from the sample data contexts
    with open(sample_path, "r") as f:
        data = json.load(f)
    pairs = data.get("qa_pairs", [])
    seen = set()
    chunks = []
    for pair in pairs:
        ctx = pair["context"]
        if ctx not in seen:
            seen.add(ctx)
            chunks.append({
                "id": len(chunks),
                "text": ctx,
                "content_type": "sample",
                "key_concepts": [],
            })

    index_path = os.path.join(tmp_dir, "chunk_index.json")
    with open(index_path, "w") as f:
        json.dump({"chunks": chunks, "total": len(chunks)}, f)

    from querySLM import ChunkRetriever
    retriever = ChunkRetriever(index_path)

    results = retriever.retrieve("What is a prime number?", top_k=3)
    if not results:
        logger.error("FAIL: No chunks retrieved")
        return False

    top = results[0]
    logger.info("Top chunk (score=%.4f): %s...", top["score"], top["text"][:80])

    if "prime" not in top["text"].lower():
        logger.warning("WARNING: Top chunk may not be the most relevant")

    logger.info("PASS: Retrieved %d chunks", len(results))
    return True


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Acamethics pipeline smoke test")
    parser.add_argument(
        "--sample-data",
        default=os.path.join(os.path.dirname(__file__), "sample_data", "sample_qa.json"),
        help="Path to sample_qa.json",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.sample_data):
        logger.error("Sample data not found: %s", args.sample_data)
        sys.exit(2)

    if not check_imports():
        sys.exit(2)

    # Add the script directory to path so we can import sibling modules
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    tmp_dir = tempfile.mkdtemp(prefix="acamethics_smoke_")
    logger.info("Temp directory: %s", tmp_dir)

    try:
        # Test 1: Schema validation
        record(test_schema_validation(args.sample_data))

        # Test 2: Fine-tuning
        model_dir = test_fine_tuning(args.sample_data, tmp_dir)
        record(bool(model_dir))

        if model_dir:
            # Test 3: Text-level evaluation
            record(test_text_evaluation(model_dir, args.sample_data))

            # Test 4: Inference
            record(test_inference(model_dir))
        else:
            logger.warning("Skipping tests 3-4 (fine-tuning failed)")
            record(False)
            record(False)

        # Test 5: Chunk retrieval
        record(test_chunk_retrieval(args.sample_data, tmp_dir))

    finally:
        logger.info("Cleaning up %s", tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary
    total = PASS_COUNT + FAIL_COUNT
    logger.info("=" * 50)
    logger.info("SMOKE TEST SUMMARY: %d/%d passed", PASS_COUNT, total)
    logger.info("=" * 50)

    if FAIL_COUNT > 0:
        logger.error("%d test(s) FAILED", FAIL_COUNT)
        sys.exit(1)
    else:
        logger.info("All tests PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()