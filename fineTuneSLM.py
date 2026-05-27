import argparse
import json
import os
import logging
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MathQADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        self.start_positions = []
        self.end_positions = []
        self.raw_pairs = []

        for qa_pair in tqdm(qa_pairs, desc="Tokenizing"):
            question = qa_pair["question"]
            context = qa_pair["context"]
            answer = qa_pair["answer"]

            # Force offset mapping with fast tokenizer
            encoding = tokenizer(
                text=question,
                text_pair=context,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            start_pos, end_pos = self._find_answer_positions(context, answer, encoding)

            if start_pos == 0 and end_pos == 0 and answer.strip():
                continue

            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            })
            self.start_positions.append(start_pos)
            self.end_positions.append(end_pos)
            self.raw_pairs.append(qa_pair)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.encodings[idx].items()}
        item["start_positions"] = self.start_positions[idx]
        item["end_positions"] = self.end_positions[idx]
        return item

    @staticmethod
    def _find_answer_positions(context, answer, encoding):
        offset_mapping = encoding["offset_mapping"][0].tolist()
        start_char = context.find(answer)
        if start_char == -1:
            return 0, 0
        end_char = start_char + len(answer) - 1

        start_token = end_token = None
        for idx, (ts, te) in enumerate(offset_mapping):
            if ts == te:
                continue
            if start_token is None and ts <= start_char < te:
                start_token = idx
            if ts <= end_char < te:
                end_token = idx
                break
        return start_token or 0, end_token or 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-t", "--tmp-dir", required=True)
    parser.add_argument("-w", "--output-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cpu")

    # Load sample data
    sample_path = os.path.join(args.input_dir, "sample_qa.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qa_pairs = data.get("qa_pairs", [])

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForQuestionAnswering.from_pretrained(model_name).to(device)

    train_pairs, val_pairs = train_test_split(qa_pairs, test_size=0.2, random_state=42)

    train_dataset = MathQADataset(train_pairs, tokenizer)
    val_dataset = MathQADataset(val_pairs, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.tmp_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting fine-tuning on %d samples...", len(train_dataset))
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training completed and model saved!")


if __name__ == "__main__":
    main()