import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import argparse
import logging
import os
import json
import math
import re
from collections import Counter
from typing import List
from flask import Flask, request, jsonify, render_template_string

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TF-IDF Chunk Retriever
# ---------------------------------------------------------------------------

class ChunkRetriever:
    """Lightweight TF-IDF retriever over a chunk index.

    Given a chunk index JSON (exported by genQA.py --export-chunks), this
    class builds an in-memory TF-IDF index and retrieves the top-k most
    relevant chunks for any query string.
    """

    def __init__(self, chunk_index_path: str):
        with open(chunk_index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        if not self.chunks:
            raise ValueError(f"No chunks found in {chunk_index_path}")

        self._tokenized = [self._tokenize(c["text"]) for c in self.chunks]
        self._idf = self._compute_idf()
        self._tfidf_vecs = [self._tfidf(tokens) for tokens in self._tokenized]
        logger.info(
            "ChunkRetriever loaded %d chunks from %s",
            len(self.chunks), chunk_index_path,
        )

    @staticmethod
    def _tokenize(text: str) -> List:
        """Simple whitespace + lowercasing tokenizer."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _compute_idf(self) -> dict:
        n = len(self._tokenized)
        df: Counter = Counter()
        for tokens in self._tokenized:
            for t in set(tokens):
                df[t] += 1
        return {t: math.log((n + 1) / (count + 1)) + 1 for t, count in df.items()}

    def _tfidf(self, tokens: list) -> dict:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {t: (count / total) * self._idf.get(t, 1.0) for t, count in tf.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Return top_k chunks most relevant to the query.

        Returns:
            list of dicts, each with 'text', 'score', and original chunk fields.
        """
        q_tokens = self._tokenize(query)
        q_vec = self._tfidf(q_tokens)
        scored = []
        for i, vec in enumerate(self._tfidf_vecs):
            score = self._cosine(q_vec, vec)
            scored.append((score, i))
        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            results.append({**self.chunks[idx], "score": round(score, 4)})
        return results


# ---------------------------------------------------------------------------
# QA Model
# ---------------------------------------------------------------------------

class SLMQuery:
    """Query a fine-tuned DistilBERT extractive QA model."""

    def __init__(self, model_dir: str, chunk_retriever: ChunkRetriever = None):
        self.model_dir = model_dir
        self.retriever = chunk_retriever
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
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
            model = DistilBertForQuestionAnswering.from_pretrained(self.model_dir)
            logger.info("Loaded model from %s", self.model_dir)
            return tokenizer, model
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise ValueError(f"Failed to load model from {self.model_dir}")

    def _extract_answer(self, question: str, context: str, max_length: int = 512) -> str:
        """Run extractive QA on a single (question, context) pair."""
        self.model.eval()
        inputs = self.tokenizer(
            question.strip(), context.strip(),
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            s = torch.argmax(outputs.start_logits, dim=1).item()
            e = torch.argmax(outputs.end_logits, dim=1).item()
        if e < s:
            e = s
        tokens = inputs["input_ids"][0][s : e + 1]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def answer(self, question: str, context: str = "", top_k: int = 3) -> dict:
        """Answer a question, optionally retrieving context automatically.

        If context is provided, it is used directly.  Otherwise, if a
        ChunkRetriever is available, the top_k most relevant chunks are
        retrieved and the best answer across them is returned.

        Returns:
            dict with 'answer', 'context_used', and optionally 'retrieved_chunks'.
        """
        if not question.strip():
            return {"answer": "No question provided.", "context_used": ""}

        # If context provided explicitly, use it
        if context.strip():
            ans = self._extract_answer(question, context)
            return {"answer": ans, "context_used": context.strip()}

        # If retriever is available, find context automatically
        if self.retriever:
            retrieved = self.retriever.retrieve(question, top_k=top_k)
            best_answer = ""
            best_context = ""
            for chunk in retrieved:
                ans = self._extract_answer(question, chunk["text"])
                if len(ans.strip()) > len(best_answer.strip()):
                    best_answer = ans
                    best_context = chunk["text"]
            return {
                "answer": best_answer if best_answer.strip() else "Could not find an answer.",
                "context_used": best_context,
                "retrieved_chunks": retrieved,
            }

        # No context and no retriever
        return {
            "answer": "No context provided and no chunk index loaded. "
                      "Please provide context or use --chunk-index.",
            "context_used": "",
        }

    def answer_batch(self, qa_items: list, top_k: int = 3) -> list:
        """Batch answer a list of items."""
        return [self.answer(item["question"], item.get("context", ""), top_k) for item in qa_items]

    def answer_from_file(self, input_file: str, output_file: str, top_k: int = 3):
        """Process questions from a file.

        Input format: one entry per line.
          - Plain question (context auto-retrieved if retriever available)
          - question<TAB>context (explicit context)
        """
        if not os.path.isfile(input_file):
            raise ValueError(f"Input file {input_file} does not exist")

        items = []
        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    q, c = line.split("\t", 1)
                    items.append({"question": q, "context": c})
                else:
                    items.append({"question": line, "context": ""})

        results = self.answer_batch(items, top_k)

        with open(output_file, "w") as f:
            for item, result in zip(items, results):
                f.write(f"Question: {item['question']}\n")
                f.write(f"Answer: {result['answer']}\n")
                if result.get("context_used"):
                    f.write(f"Context: {result['context_used'][:200]}...\n")
                f.write("\n")

        logger.info("Answers saved to %s", output_file)


# ---------------------------------------------------------------------------
# CLI / Server
# ---------------------------------------------------------------------------

def run_cli(args):
    retriever = ChunkRetriever(args.chunk_index) if args.chunk_index else None
    slm = SLMQuery(args.model_dir, chunk_retriever=retriever)
    slm.answer_from_file(args.input, args.output)


def run_server(args):
    retriever = ChunkRetriever(args.chunk_index) if args.chunk_index else None
    slm = SLMQuery(args.model_dir, chunk_retriever=retriever)
    has_retriever = retriever is not None
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok", "model_dir": args.model_dir,
            "device": str(slm.device),
            "chunk_retrieval": has_retriever,
        })

    @app.route("/query", methods=["POST"])
    def query():
        data = request.json
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question'"}), 400
        q = str(data["question"])[:1000]
        c = str(data.get("context", ""))[:5000]
        result = slm.answer(q, c)
        return jsonify({"question": q, **result})

    @app.route("/", methods=["GET", "POST"])
    def index():
        question = context = answer = context_used = None
        retrieved = []
        if request.method == "POST":
            question = request.form.get("question", "")[:1000]
            context = request.form.get("context", "")[:5000]
            if question:
                result = slm.answer(question, context)
                answer = result["answer"]
                context_used = result.get("context_used", "")
                retrieved = result.get("retrieved_chunks", [])
        return render_template_string(
            HTML_TEMPLATE,
            question=question, context=context, answer=answer,
            context_used=context_used, retrieved=retrieved,
            has_retriever=has_retriever,
        )

    logger.info("Starting server on port %d (retrieval=%s)", args.port, has_retriever)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Math QA</title>
    <style>
        body { font-family: sans-serif; max-width: 750px; margin: 40px auto; padding: 0 20px; }
        textarea, input[type=text] { width: 100%; padding: 8px; margin: 4px 0 12px 0; box-sizing: border-box; }
        input[type=submit] { padding: 8px 24px; cursor: pointer; }
        .result { background: #f4f4f4; padding: 16px; border-radius: 4px; margin-top: 16px; }
        .chunk { background: #eef; padding: 10px; border-radius: 4px; margin: 6px 0; font-size: 0.9em; }
        .note { color: #666; font-size: 0.85em; margin-bottom: 12px; }
    </style>
</head>
<body>
    <h1>Math QA</h1>
    {% if has_retriever %}
    <p class="note">Chunk retrieval is enabled. You can leave the context field empty and
    the system will automatically find the most relevant passage.</p>
    {% else %}
    <p class="note">No chunk index loaded. Provide context below, or restart the server
    with --chunk-index to enable automatic retrieval.</p>
    {% endif %}
    <form method="post">
        <label for="question">Question:</label>
        <input type="text" id="question" name="question" placeholder="e.g. What is a prime number?"
               value="{{ question or '' }}">
        <label for="context">Context (optional if retrieval is enabled):</label>
        <textarea id="context" name="context" rows="4"
                  placeholder="Paste the relevant text here, or leave empty for auto-retrieval...">{{ context or '' }}</textarea>
        <input type="submit" value="Submit">
    </form>
    {% if answer %}
    <div class="result">
        <h2>Answer:</h2>
        <p>{{ answer }}</p>
        {% if context_used %}
        <h3>Context used:</h3>
        <p style="font-size:0.9em; color:#444;">{{ context_used[:500] }}{% if context_used|length > 500 %}...{% endif %}</p>
        {% endif %}
        {% if retrieved %}
        <h3>Retrieved chunks:</h3>
        {% for chunk in retrieved %}
        <div class="chunk">
            <strong>Score: {{ chunk.score }}</strong> ({{ chunk.content_type }})<br>
            {{ chunk.text[:300] }}{% if chunk.text|length > 300 %}...{% endif %}
        </div>
        {% endfor %}
        {% endif %}
    </div>
    {% endif %}
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a fine-tuned DistilBERT model for math QA.")
    parser.add_argument("-m", "--model_dir", required=True, help="Directory containing the fine-tuned model.")
    parser.add_argument("--mode", choices=["cli", "server"], default="cli")
    parser.add_argument("--chunk-index", default=None,
                        help="Path to chunk index JSON (from genQA.py --export-chunks). "
                             "Enables automatic context retrieval.")
    # CLI
    parser.add_argument("-i", "--input", help="Input file with questions (CLI mode).")
    parser.add_argument("-o", "--output", help="Output file for answers (CLI mode).")
    # Server
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    if args.mode == "cli":
        if not args.input or not args.output:
            parser.error("CLI mode requires --input and --output.")
        run_cli(args)
    elif args.mode == "server":
        run_server(args)