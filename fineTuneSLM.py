import argparse
import json
import os
from transformers import (
    DistilBertTokenizer,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Custom Dataset for QA pairs
class MathQADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=512):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        context = qa_pair["context"]

        # Tokenize inputs
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Find start and end positions of the answer in the tokenized context
        answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        context_tokens = encoding["input_ids"][0]

        start_positions = []
        end_positions = []

        # Search for answer tokens in context
        for i in range(len(context_tokens) - len(answer_tokens) + 1):
            if context_tokens[i : i + len(answer_tokens)].tolist() == answer_tokens:
                start_positions.append(i)
                end_positions.append(i + len(answer_tokens) - 1)
                break
        else:
            # If answer not found, use default positions (CLS token)
            start_positions.append(0)
            end_positions.append(0)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "start_positions": start_positions[0] if start_positions else 0,
            "end_positions": end_positions[0] if end_positions else 0,
        }

def load_qa_pairs(input_dir):
    qa_pairs = []
    for filename in tqdm(os.listdir(input_dir), desc="Loading JSON files"):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                qa_pairs.extend(data.get("qa_pairs", []))
    return qa_pairs

def main():
    parser = argparse.ArgumentParser(description="Fine-tune an SLM for 6th-grade math QA")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing JSON files")
    parser.add_argument("-t", "--tmp-dir", required=True, help="Directory for saving checkpoints")
    parser.add_argument("-w", "--output-dir", required=True, help="Directory to save final model")
    args = parser.parse_args()

    # Check if MPS is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForQuestionAnswering.from_pretrained(model_name).to(device)

    # Load QA pairs
    qa_pairs = load_qa_pairs(args.input_dir)
    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Create dataset
    dataset = MathQADataset(qa_pairs, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.tmp_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Adjusted for M4 with 32GB RAM
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(args.tmp_dir, "logs"),
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=False,  # MPS does not support fp16; use full precision
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune the model
    print("Starting fine-tuning...")
    trainer.train()

    # Save the final model
    print(f"Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()