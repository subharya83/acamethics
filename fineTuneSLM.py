import os
import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load the dataset from the directory
def load_dataset(data_dir):
    qa_pairs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r") as f:
                qa_pairs.extend(json.load(f))
    return qa_pairs

# Preprocess the dataset for fine-tuning
def preprocess_dataset(qa_pairs, tokenizer, max_length=512):
    inputs = []
    targets = []
    for pair in qa_pairs:
        inputs.append(f"generate questions: {pair['context']}")
        targets.append(pair["question"] + " " + pair["answer"])  # Combine question and answer

    # Tokenize the inputs and targets
    tokenized_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    tokenized_targets = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }

# Fine-tune the model
def fine_tune_model(data_dir, output_dir, model_name="t5-small", epochs=3, batch_size=8):
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load and preprocess the dataset
    qa_pairs = load_dataset(data_dir)
    processed_data = preprocess_dataset(qa_pairs, tokenizer)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(processed_data)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="no",  # No evaluation during training
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune the model
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Fine-tune a small language model for math QA.")
    parser.add_argument("-d", "--data_dir", required=True, help="Directory containing QA pairs in JSON format.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("-m", "--model_name", default="t5-small", help="Name of the pre-trained model to fine-tune.")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Training batch size.")
    args = parser.parse_args()

    # Run the fine-tuning process
    fine_tune_model(args.data_dir, args.output_dir, args.model_name, args.epochs, args.batch_size)
    