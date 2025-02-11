import torch
from transformers import pipeline
import pdfplumber
import json

# Load an open-source LLM for question-answering
def load_qa_model():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    qa_pipeline = pipeline("question-generation", model="valhalla/t5-small-qa-qg-hl", device=device)
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
        result = qa_pipeline(chunk)
        qa_pairs.extend(result)

    return qa_pairs

# Save QA pairs to a JSON file
def save_qa_pairs(qa_pairs, output_file):
    with open(output_file, "w") as f:
        json.dump(qa_pairs, f, indent=4)

# Main function
def main(pdf_path, output_file):
    # Step 1: Load the QA model
    qa_pipeline = load_qa_model()

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
    pdf_path = "input.pdf"  # Replace with your PDF file path
    output_file = "output_qa_pairs.json"  # Output file name
    main(pdf_path, output_file)