import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import logging
import os
from flask import Flask, request, jsonify, render_template_string
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SLMQuery:
    """Class for querying a fine-tuned T5 model for question answering."""
    
    def __init__(self, model_dir):
        """Initialize the SLMQuery with model and tokenizer.
        
        Args:
            model_dir (str): Directory containing the fine-tuned model and tokenizer.
        """
        self.model_dir = model_dir
        self.tokenizer, self.model = self.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def load_model(self):
        """Load the tokenizer and model from the specified directory.
        
        Returns:
            tuple: Tokenizer and model instances.
        """
        try:
            tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
            model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
            logger.info(f"Successfully loaded model and tokenizer from {self.model_dir}")
            return tokenizer, model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Failed to load model from {self.model_dir}")

    def generate_answer(self, question, max_length=512, num_beams=4):
        """Generate an answer for a single question.
        
        Args:
            question (str): The question to answer.
            max_length (int): Maximum length of the generated answer.
            num_beams (int): Number of beams for beam search.
        
        Returns:
            str: The generated answer.
        """
        if not question.strip():
            logger.warning("Empty question provided")
            return "No question provided."

        input_text = f"answer question: {question.strip()}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Generated answer for question: '{question.strip()}'")
        return answer

    def generate_answers_from_file(self, input_file, output_file):
        """Generate answers for questions in an input file and save to output file.
        
        Args:
            input_file (str): Path to the input text file containing questions.
            output_file (str): Path to the output text file for answers.
        """
        if not os.path.isfile(input_file):
            raise ValueError(f"Input file {input_file} does not exist")

        with open(input_file, "r") as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]

        answers = [self.generate_answer(q) for q in questions]

        with open(output_file, "w") as f:
            for question, answer in zip(questions, answers):
                f.write(f"Question: {question}\nAnswer: {answer}\n\n")

        logger.info(f"Answers saved to {output_file}")

def run_cli(args):
    """Run in CLI mode to process questions from a file."""
    query_slm = SLMQuery(args.model_dir)
    query_slm.generate_answers_from_file(args.input, args.output)

def run_server(args):
    """Run in server mode with a Flask web server and minimal GUI."""
    query_slm = SLMQuery(args.model_dir)
    app = Flask(__name__)

    @app.route('/query', methods=['POST'])
    def query():
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request"}), 400
        question = data['question']
        answer = query_slm.generate_answer(question)
        return jsonify({"question": question, "answer": answer})

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            question = request.form.get('question')
            if question:
                answer = query_slm.generate_answer(question)
                return render_template_string(HTML_TEMPLATE, question=question, answer=answer)
        return render_template_string(HTML_TEMPLATE, question=None, answer=None)

    logger.info(f"Starting web server on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)

# Minimal GUI HTML template
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SLM Query GUI</title>
</head>
<body>
    <h1>Query the SLM Model</h1>
    <form method="post">
        <label for="question">Enter your question:</label><br>
        <input type="text" id="question" name="question" style="width: 400px;"><br><br>
        <input type="submit" value="Submit">
    </form>
    {% if question %}
    <h2>Question:</h2>
    <p>{{ question }}</p>
    <h2>Answer:</h2>
    <p>{{ answer }}</p>
    {% endif %}
</body>
</html>
"""

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Query a fine-tuned T5 model for answers.")
    parser.add_argument("-m", "--model_dir", required=True, help="Directory containing the fine-tuned model.")
    parser.add_argument("--mode", choices=['cli', 'server'], default='cli', help="Run mode: 'cli' for command-line or 'server' for web server.")
    
    # CLI-specific arguments
    parser.add_argument("-i", "--input", help="Path to the input text file containing questions (required for CLI mode).")
    parser.add_argument("-o", "--output", help="Path to the output text file for answers (required for CLI mode).")
    
    # Server-specific arguments
    parser.add_argument("--port", type=int, default=5000, help="Port for the web server (default: 5000).")
    
    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'cli':
        if not args.input or not args.output:
            parser.error("CLI mode requires --input and --output arguments.")
        run_cli(args)
    elif args.mode == 'server':
        run_server(args)