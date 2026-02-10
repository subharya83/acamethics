import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import argparse
import logging
import os
from flask import Flask, request, jsonify, render_template_string

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SLMQuery:
    """Class for querying a fine-tuned DistilBERT model for extractive QA.

    The original code loaded a T5 (generative) model, but fineTuneSLM.py
    produces a DistilBERT (extractive) model.  This version matches the
    fine-tuning architecture so saved weights load correctly.
    """

    def __init__(self, model_dir):
        """Initialize with model and tokenizer.

        Args:
            model_dir (str): Directory containing the fine-tuned model.
        """
        self.model_dir = model_dir
        self.tokenizer, self.model = self._load_model()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        logger.info("Model loaded on device: %s", self.device)

    def _load_model(self):
        """Load the tokenizer and model from the specified directory.

        Returns:
            tuple: (tokenizer, model)
        """
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
            model = DistilBertForQuestionAnswering.from_pretrained(self.model_dir)
            logger.info(
                "Successfully loaded model and tokenizer from %s",
                self.model_dir,
            )
            return tokenizer, model
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise ValueError(f"Failed to load model from {self.model_dir}")

    def generate_answer(self, question, context="", max_length=512):
        """Generate an answer for a question given optional context.

        For extractive QA the caller should supply context. If no context
        is provided, the question itself is used as context (which will
        produce poor results but avoids crashing).

        Args:
            question (str): The question to answer.
            context (str): The context passage to extract the answer from.
            max_length (int): Maximum sequence length.

        Returns:
            str: The extracted answer span.
        """
        if not question.strip():
            logger.warning("Empty question provided")
            return "No question provided."

        if not context.strip():
            context = question

        self.model.eval()
        inputs = self.tokenizer(
            question.strip(),
            context.strip(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            start_pos = torch.argmax(outputs.start_logits, dim=1).item()
            end_pos = torch.argmax(outputs.end_logits, dim=1).item()

        if end_pos < start_pos:
            end_pos = start_pos

        answer_tokens = inputs["input_ids"][0][start_pos : end_pos + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        logger.info("Generated answer for question: '%s'", question.strip())
        return answer

    def generate_answers_batch(self, qa_items, max_length=512, batch_size=8):
        """Batch inference for a list of (question, context) pairs.

        Args:
            qa_items (list): List of dicts with 'question' and 'context'.
            max_length (int): Maximum sequence length.
            batch_size (int): Batch size for inference.

        Returns:
            list: List of answer strings.
        """
        self.model.eval()
        answers = []

        for i in range(0, len(qa_items), batch_size):
            batch = qa_items[i : i + batch_size]
            questions = [item["question"].strip() for item in batch]
            contexts = [item.get("context", item["question"]).strip() for item in batch]

            inputs = self.tokenizer(
                questions,
                contexts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                start_positions = torch.argmax(outputs.start_logits, dim=1)
                end_positions = torch.argmax(outputs.end_logits, dim=1)

            for j in range(len(batch)):
                s = start_positions[j].item()
                e = end_positions[j].item()
                if e < s:
                    e = s
                tokens = inputs["input_ids"][j][s : e + 1]
                answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
                answers.append(answer)

        return answers

    def generate_answers_from_file(self, input_file, output_file):
        """Generate answers for questions in an input file.

        Each line in the input file should be formatted as:
            question [TAB] context
        If no tab is present, the question is used as its own context.

        Args:
            input_file (str): Path to input text file.
            output_file (str): Path to output text file.
        """
        if not os.path.isfile(input_file):
            raise ValueError(f"Input file {input_file} does not exist")

        qa_items = []
        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    question, context = line.split("\t", 1)
                else:
                    question = line
                    context = line
                qa_items.append({"question": question, "context": context})

        answers = self.generate_answers_batch(qa_items)

        with open(output_file, "w") as f:
            for item, answer in zip(qa_items, answers):
                f.write(f"Question: {item['question']}\nAnswer: {answer}\n\n")

        logger.info("Answers saved to %s", output_file)


def run_cli(args):
    """Run in CLI mode to process questions from a file."""
    query_slm = SLMQuery(args.model_dir)
    query_slm.generate_answers_from_file(args.input, args.output)


def run_server(args):
    """Run in server mode with a Flask web server and minimal GUI."""
    query_slm = SLMQuery(args.model_dir)
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify(
            {
                "status": "ok",
                "model_dir": args.model_dir,
                "device": str(query_slm.device),
            }
        )

    @app.route("/query", methods=["POST"])
    def query():
        data = request.json
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request"}), 400

        question = str(data["question"])[:1000]  # limit length
        context = str(data.get("context", ""))[:5000]
        answer = query_slm.generate_answer(question, context)
        return jsonify({"question": question, "answer": answer})

    @app.route("/", methods=["GET", "POST"])
    def index():
        question = None
        context = None
        answer = None
        if request.method == "POST":
            question = request.form.get("question", "")[:1000]
            context = request.form.get("context", "")[:5000]
            if question:
                answer = query_slm.generate_answer(question, context)
        return render_template_string(
            HTML_TEMPLATE, question=question, context=context, answer=answer
        )

    logger.info("Starting web server on port %d", args.port)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


# Minimal GUI HTML template
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Math QA - SLM Query</title>
    <style>
        body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
        textarea, input[type=text] { width: 100%; padding: 8px; margin: 4px 0 12px 0; box-sizing: border-box; }
        input[type=submit] { padding: 8px 24px; cursor: pointer; }
        .result { background: #f4f4f4; padding: 16px; border-radius: 4px; margin-top: 16px; }
    </style>
</head>
<body>
    <h1>Math QA</h1>
    <form method="post">
        <label for="question">Question:</label>
        <input type="text" id="question" name="question" placeholder="e.g. What is 2 + 2?"
               value="{{ question or '' }}">
        <label for="context">Context (passage to search for the answer):</label>
        <textarea id="context" name="context" rows="4"
                  placeholder="Paste the relevant text here...">{{ context or '' }}</textarea>
        <input type="submit" value="Submit">
    </form>
    {% if answer %}
    <div class="result">
        <h2>Question:</h2>
        <p>{{ question }}</p>
        <h2>Answer:</h2>
        <p>{{ answer }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a fine-tuned DistilBERT model for math QA."
    )
    parser.add_argument(
        "-m", "--model_dir", required=True,
        help="Directory containing the fine-tuned model.",
    )
    parser.add_argument(
        "--mode", choices=["cli", "server"], default="cli",
        help="Run mode: 'cli' for command-line or 'server' for web server.",
    )

    # CLI-specific arguments
    parser.add_argument(
        "-i", "--input",
        help="Path to input text file containing questions (required for CLI mode).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output text file for answers (required for CLI mode).",
    )

    # Server-specific arguments
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port for the web server (default: 5000).",
    )

    args = parser.parse_args()

    if args.mode == "cli":
        if not args.input or not args.output:
            parser.error("CLI mode requires --input and --output arguments.")
        run_cli(args)
    elif args.mode == "server":
        run_server(args)