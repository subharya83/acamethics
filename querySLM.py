import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse

# Load the fine-tuned model and tokenizer
def load_model(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return tokenizer, model

# Generate answers for questions in the input file
def generate_answers(input_file, output_file, model_dir):
    # Load the model and tokenizer
    tokenizer, model = load_model(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Read questions from the input file
    with open(input_file, "r") as f:
        questions = f.readlines()

    # Generate answers for each question
    answers = []
    for question in questions:
        # Preprocess the question
        input_text = f"answer question: {question.strip()}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        # Generate the answer
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)

        # Decode the output
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answers.append(answer)

    # Save the questions and answers to the output file
    with open(output_file, "w") as f:
        for question, answer in zip(questions, answers):
            f.write(f"Question: {question.strip()}\nAnswer: {answer}\n\n")

    print(f"Answers saved to {output_file}.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate answers using a fine-tuned model.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input text file containing questions.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output text file for answers.")
    parser.add_argument("-m", "--model_dir", required=True, help="Directory containing the fine-tuned model.")
    args = parser.parse_args()

    # Run the inference
    generate_answers(args.input, args.output, args.model_dir)