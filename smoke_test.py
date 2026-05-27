#!/usr/bin/env python3
"""End-to-end smoke test for Acamethics"""

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
    required = ["torch", "transformers", "sklearn", "tqdm"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        logger.error("Missing packages: %s", missing)
        return False
    return True


def test_schema_validation(sample_path: str) -> bool:
    logger.info("=" * 60)
    logger.info("TEST 1: Schema Validation")
    logger.info("=" * 60)

    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("qa_pairs", [])
    logger.info("PASS: %d QA pairs validated", len(pairs))
    return True


def test_fine_tuning(sample_path: str, tmp_dir: str) -> str:
    logger.info("=" * 60)
    logger.info("TEST 2: Fine-tuning (1 epoch)")
    logger.info("=" * 60)

    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(sample_path, os.path.join(data_dir, "sample_qa.json"))

    output_dir = os.path.join(tmp_dir, "model")
    ckpt_dir = os.path.join(tmp_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Run fine-tuning using the simplified script
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, "fineTuneSLM.py",
            "-i", data_dir,
            "-t", ckpt_dir,
            "-w", output_dir,
            "--num-epochs", "1"
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))

        if result.returncode == 0:
            logger.info("PASS: Fine-tuning completed successfully")
            logger.info(result.stdout[-500:])  # Show last part of output
            return output_dir
        else:
            logger.error("Fine-tuning failed:\n%s", result.stderr)
            return ""
    except Exception as e:
        logger.error("Error running fineTuneSLM.py: %s", e)
        return ""


def test_inference(model_dir: str) -> bool:
    logger.info("=" * 60)
    logger.info("TEST 3: Inference Test")
    logger.info("=" * 60)

    try:
        from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
        import torch

        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model = DistilBertForQuestionAnswering.from_pretrained(model_dir)
        device = torch.device("cpu")
        model.to(device)

        # Quick test
        question = "What is a prime number?"
        context = "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."
        
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            s = torch.argmax(outputs.start_logits, dim=1).item()
            e = torch.argmax(outputs.end_logits, dim=1).item()
        
        answer = tokenizer.decode(inputs["input_ids"][0][s:e+1], skip_special_tokens=True)
        logger.info("Question: %s", question)
        logger.info("Answer : %s", answer)
        logger.info("PASS: Inference successful")
        return True
    except Exception as e:
        logger.error("Inference test failed: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-data", default="sample_data/sample_qa.json")
    args = parser.parse_args()

    sample_path = os.path.join(os.path.dirname(__file__), args.sample_data)

    if not os.path.exists(sample_path):
        logger.error("Sample data not found: %s", sample_path)
        sys.exit(2)

    if not check_imports():
        sys.exit(2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    tmp_dir = tempfile.mkdtemp(prefix="acamethics_smoke_")
    logger.info("Temp directory: %s", tmp_dir)

    try:
        record(test_schema_validation(sample_path))
        model_dir = test_fine_tuning(sample_path, tmp_dir)
        record(bool(model_dir))

        if model_dir:
            record(test_inference(model_dir))
        else:
            record(False)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total = PASS_COUNT + FAIL_COUNT
    logger.info("=" * 60)
    logger.info("SMOKE TEST SUMMARY: %d/%d passed", PASS_COUNT, total)
    logger.info("=" * 60)

    if FAIL_COUNT == 0:
        logger.info("🎉 All tests PASSED!")
        sys.exit(0)
    else:
        logger.error("Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()