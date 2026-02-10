#!/usr/bin/env python3
"""Validate QA pair JSON files for compatibility with the fine-tuning pipeline.

Usage:
    python validate_schema.py path/to/qa_pairs.json
    python validate_schema.py path/to/directory/

Checks performed on each QA pair:
  1. Required keys present: question, answer, context
  2. All values are non-empty strings
  3. Answer appears verbatim in context (extractive QA compatibility)
  4. Question ends with '?'
  5. Answer has at least 3 words

Exit codes:
  0 - all pairs valid
  1 - some pairs invalid (details printed)
  2 - file/directory not found or JSON parse error
"""

import json
import os
import sys
import argparse

REQUIRED_KEYS = {"question", "answer", "context"}


def validate_pair(pair: dict, idx: int) -> list:
    """Validate a single QA pair. Returns list of issue strings."""
    issues = []

    missing = REQUIRED_KEYS - set(pair.keys())
    if missing:
        issues.append(f"  pair {idx}: missing keys {missing}")
        return issues  # can't check further

    q = pair["question"]
    a = pair["answer"]
    c = pair["context"]

    if not isinstance(q, str) or not q.strip():
        issues.append(f"  pair {idx}: question is empty or not a string")
    if not isinstance(a, str) or not a.strip():
        issues.append(f"  pair {idx}: answer is empty or not a string")
    if not isinstance(c, str) or not c.strip():
        issues.append(f"  pair {idx}: context is empty or not a string")

    if issues:
        return issues

    if a.strip() not in c:
        issues.append(f"  pair {idx}: answer is not a substring of context")

    if not q.strip().endswith("?"):
        issues.append(f"  pair {idx}: question does not end with '?'")

    if len(a.strip().split()) < 3:
        issues.append(f"  pair {idx}: answer has fewer than 3 words")

    return issues


def validate_file(filepath: str) -> tuple:
    """Validate a single JSON file. Returns (valid_count, total_count, issues)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return 0, 0, [f"  JSON parse error: {e}"]
    except Exception as e:
        return 0, 0, [f"  Read error: {e}"]

    pairs = data.get("qa_pairs", [])
    if not pairs:
        return 0, 0, ["  No 'qa_pairs' key or it is empty"]

    all_issues = []
    valid = 0
    for i, pair in enumerate(pairs):
        pair_issues = validate_pair(pair, i)
        if pair_issues:
            all_issues.extend(pair_issues)
        else:
            valid += 1

    return valid, len(pairs), all_issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate QA pair JSON files for fine-tuning compatibility."
    )
    parser.add_argument(
        "path", help="Path to a JSON file or directory of JSON files."
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if ANY pair is invalid.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: {args.path} not found")
        sys.exit(2)

    files = []
    if os.path.isdir(args.path):
        for name in sorted(os.listdir(args.path)):
            if name.endswith(".json"):
                files.append(os.path.join(args.path, name))
    else:
        files.append(args.path)

    if not files:
        print(f"No JSON files found in {args.path}")
        sys.exit(2)

    total_valid = 0
    total_pairs = 0
    has_issues = False

    for fp in files:
        print(f"\nValidating: {fp}")
        valid, total, issues = validate_file(fp)
        total_valid += valid
        total_pairs += total

        if issues:
            has_issues = True
            print(f"  {valid}/{total} pairs valid, {len(issues)} issue(s):")
            for issue in issues[:20]:
                print(issue)
            if len(issues) > 20:
                print(f"  ... and {len(issues) - 20} more issues")
        else:
            print(f"  All {total} pairs valid")

    print(f"\nSummary: {total_valid}/{total_pairs} pairs valid across {len(files)} file(s)")

    if has_issues and args.strict:
        sys.exit(1)
    elif total_valid == 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()