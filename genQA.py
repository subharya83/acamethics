import torch
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import pdfplumber
import json
import os
import argparse

# Download and load the QA model locally
def load_qa_model(weights_dir="weights"):
    # Create the weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Load the model and tokenizer
    model_name = "valhalla/t5-small-qa-qg-hl"
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=weights_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=weights_dir)

    # Set up the QA pipeline
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return qa_pipeline

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Generate question-answer pairs from the text
def generate_qa_pairs(text, qa_pipeline, max_length=512):
    # Split text into chunks to fit within the model's token limit
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    qa_pairs = []

    for chunk in chunks:
        # Generate QA pairs for each chunk
        result = qa_pipeline(f"generate questions: {chunk}")
        qa_pairs.extend(result)

    return qa_pairs

# Save QA pairs to a JSON file
def save_qa_pairs(qa_pairs, output_file):
    with open(output_file, "w") as f:
        json.dump(qa_pairs, f, indent=4)

# Main function
def main(pdf_path, output_file, weights_dir="weights"):
    # Step 1: Load the QA model
    qa_pipeline = load_qa_model(weights_dir)
    print("QA model loaded.")

    # Step 2: Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    print("Text extracted from PDF.")

    # Step 3: Generate QA pairs
    qa_pairs = generate_qa_pairs(text, qa_pipeline)
    print("QA pairs generated.")

    # Step 4: Save QA pairs to a file
    save_qa_pairs(qa_pairs, output_file)
    print(f"QA pairs saved to {output_file}.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate QA pairs from a PDF file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file.")
    parser.add_argument("-w", "--weights", default="weights", help="Directory to store model weights.")
    args = parser.parse_args()

    # Run the main function
    main(args.input, args.output, args.weights)