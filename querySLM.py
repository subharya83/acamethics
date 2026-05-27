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
    """Lightweight TF-IDF retriever over a chunk index."""

    def __init__(self, chunk_index_path: str):
        with open(chunk_index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        if not self.chunks:
            raise ValueError(f"No chunks found in {chunk_index_path}")

        self._tokenized = [self._tokenize(c["text"]) for c in self.chunks]
        self._idf = self._compute_idf()
        self._tfidf_vecs = [self._tfidf(tokens) for tokens in self._tokenized]
        logger.info("ChunkRetriever loaded %d chunks", len(self.chunks))

    @staticmethod
    def _tokenize(text: str) -> List:
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
            # FIXED: Use fast tokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(
                self.model_dir, 
                use_fast=True
            )
            model = DistilBertForQuestionAnswering.from_pretrained(self.model_dir)
            logger.info("Loaded model from %s", self.model_dir)
            return tokenizer, model
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise ValueError(f"Failed to load model from {self.model_dir}")

    def _extract_answer(self, question: str, context: str, max_length: int = 512) -> str:
        self.model.eval()
        inputs = self.tokenizer(
            question.strip(), context.strip(),
            max_length=max_length, 
            padding="max_length",
            truncation=True, 
            return_tensors="pt",
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
        if not question.strip():
            return {"answer": "No question provided.", "context_used": ""}

        if context.strip():
            ans = self._extract_answer(question, context)
            return {"answer": ans, "context_used": context.strip()}

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

        return {
            "answer": "No context provided and no chunk index loaded.",
            "context_used": "",
        }

    def answer_batch(self, qa_items: list, top_k: int = 3) -> list:
        return [self.answer(item["question"], item.get("context", ""), top_k) for item in qa_items]

    def answer_from_file(self, input_file: str, output_file: str, top_k: int = 3):
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
            "status": "ok", 
            "model_dir": args.model_dir,
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

    logger.info("Starting server on port %d", args.port)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Math QA - Acamethics</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        input, textarea { width: 100%; padding: 10px; margin: 8px 0; }
        input[type=submit] { padding: 12px 30px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        .result { background: #f0f0f0; padding: 20px; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Math QA System</h1>
    <form method="post">
        <label>Question:</label><br>
        <input type="text" name="question" placeholder="Ask a math question..." value="{{ question or '' }}">
        <label>Context (optional):</label><br>
        <textarea name="context" rows="4" placeholder="Paste context here...">{{ context or '' }}</textarea>
        <input type="submit" value="Get Answer">
    </form>
    {% if answer %}
    <div class="result">
        <h2>Answer:</h2>
        <p>{{ answer }}</p>
        {% if context_used %}
        <h3>Context Used:</h3>
        <p style="font-size:0.9em;">{{ context_used[:400] }}...</p>
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
    parser.add_argument("--chunk-index", default=None, help="Path to chunk index JSON")
    parser.add_argument("-i", "--input", help="Input file (CLI mode)")
    parser.add_argument("-o", "--output", help="Output file (CLI mode)")
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    if args.mode == "cli":
        if not args.input or not args.output:
            parser.error("CLI mode requires --input and --output.")
        run_cli(args)
    elif args.mode == "server":
        run_server(args)