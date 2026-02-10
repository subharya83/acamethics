import torch
from transformers import (
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
import pdfplumber
import json
import os
import argparse
import re
import logging
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Processing
# ---------------------------------------------------------------------------

class TextProcessor:
    """Handles all text cleaning and preprocessing."""

    def __init__(self):
        self.encoding_fixes = {
            "\u00c3\u00a2\u00e2\u201a\u00ac\u00e2\u201e\u00a2": "'",
            "\u00c3\u00a2\u00e2\u201a\u00ac\u00c5\u201c": '"',
            "\u00c3\u00a2\u00e2\u201a\u00ac\u00ef\u00bf\u00bd": '"',
            "\u00c3\u00a2\u00e2\u201a\u00ac\u201c": "\u2014",
            "\u00c3\u00a2\u00e2\u201a\u00ac\u00c2\u00a6": "...",
            "\u00c3\u00a2\u00e2\u201a\u00ac\u00cb\u0153": "'",
            "\u00c3\u0082": " ",
            "\u00c3\u0083\u00c2\u00a1": "\u00e1",
            "\u00c3\u0083\u00c2\u00a9": "\u00e9",
            "\u00c3\u0082 ": " ",
            "\u00c3\u0082\u00c2\u00ad": "-",
        }
        self.noise_patterns = [
            r"Chapter \d+_.*?\.indd \d+.*?\d{2}:\d{2}:\d{2}",
            r"Ganita Prakash \| Grade \d+",
            r"Patterns in Mathematics\s*\d+",
            r"Math\s+Talk\s*",
            r"Try\s+This\s*",
            r"Figure it Out\s*",
            r"^\d+$",
            r"^\s*\.\s*\.\s*\.\s*$",
        ]

    def fix_encoding(self, text: str) -> str:
        for bad, good in self.encoding_fixes.items():
            text = text.replace(bad, good)
        return text

    def remove_noise(self, text: str) -> str:
        for pattern in self.noise_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()

    def preserve_math_sequences(self, text: str) -> str:
        seq_pattern = r"(\d+(?:,\s*\d+){3,}(?:,\s*\.{3})?)"
        sequences = re.findall(seq_pattern, text)
        for idx, seq in enumerate(sequences):
            text = text.replace(seq, f"[MATH_SEQ_{idx}:{seq}]", 1)
        return text

    def restore_math_sequences(self, text: str) -> str:
        return re.sub(r"\[MATH_SEQ_\d+:(.*?)\]", r"\1", text)

    def clean_extracted_text(self, text: str) -> str:
        text = self.fix_encoding(text)
        text = self.remove_noise(text)
        text = self.preserve_math_sequences(text)
        return text


# ---------------------------------------------------------------------------
# Content Analysis
# ---------------------------------------------------------------------------

class ContentAnalyzer:
    """Analyzes content type and extracts key information."""

    def __init__(self):
        self.text_indicators: Dict[str, List[str]] = {
            "definition": ["is defined as", "refers to", "means", "what is", "called"],
            "example": ["for example", "such as", "including", "like", "instance"],
            "sequence": ["sequence", "pattern", "next", "series"],
            "explanation": ["because", "due to", "reason", "why", "how", "explains"],
            "instruction": ["draw", "copy", "find", "calculate", "solve", "can you"],
            "mathematical": ["numbers", "triangular", "square", "cube"],
        }
        self.regex_indicators: Dict[str, List[re.Pattern]] = {
            "mathematical": [re.compile(r"\d+(?:,\s*\d+){2,}")],
        }

    def identify_content_type(self, text: str) -> str:
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for content_type, indicators in self.text_indicators.items():
            score = sum(1 for ind in indicators if ind in text_lower)
            scores[content_type] = score
        for content_type, patterns in self.regex_indicators.items():
            for pat in patterns:
                if pat.search(text_lower):
                    scores[content_type] = scores.get(content_type, 0) + 2
        return max(scores, key=scores.get) if scores else "general"

    def extract_key_concepts(self, text: str) -> List[str]:
        concepts: List[str] = []
        math_terms = re.findall(
            r"\b(?:triangular|square|cube|prime|even|odd|fibonacci|"
            r"sequence|pattern|polygon|ratio|fraction|decimal|percent|"
            r"equation|variable|exponent|factor|multiple|angle|"
            r"perimeter|area|volume|integer|rational)\b",
            text.lower(),
        )
        concepts.extend(math_terms)
        sequences = re.findall(r"\d+(?:,\s*\d+){2,}", text)
        concepts.extend([f"sequence_{seq.replace(' ', '')}" for seq in sequences])
        cap_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        concepts.extend([t.lower() for t in cap_terms if len(t.split()) <= 3])
        return list(set(concepts))


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class QuestionValidator:
    def __init__(self):
        self.min_length = 15
        self.max_length = 200
        self.question_starters = [
            "what", "how", "why", "when", "where", "which", "who",
            "can", "do", "does", "is", "are", "if", "find", "calculate", "solve",
        ]

    def is_valid_question(self, question: str) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not question or not question.strip():
            return False, ["Empty question"]
        question = question.strip()
        if len(question) < self.min_length:
            issues.append("Too short")
        elif len(question) > self.max_length:
            issues.append("Too long")
        if not question.endswith("?"):
            issues.append("Missing question mark")
        first_word = question.split()[0].lower() if question.split() else ""
        if first_word not in self.question_starters and not any(
            s in question.lower()[:20] for s in self.question_starters
        ):
            issues.append("Doesn't start like a question")
        if question.count(".") >= question.count("?"):
            issues.append("Contains statement fragments")
        words = question.lower().split()
        if len(words) > 3 and len(set(words)) < len(words) * 0.6:
            issues.append("Too repetitive")
        return len(issues) == 0, issues


class AnswerValidator:
    def __init__(self):
        self.min_length = 5
        self.max_length = 300
        self.noise_patterns = [
            r"\d{2}:\d{2}:\d{2}", r"Chapter \d+", r"\.indd",
            r"^[A-Z\s]{5,}$", r"^\d+$", r"^[^a-zA-Z0-9]*$",
        ]

    def is_valid_answer(self, answer: str, question: str = "") -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not answer or not answer.strip():
            return False, ["Empty answer"]
        answer = answer.strip()
        if len(answer) < self.min_length:
            issues.append("Too short")
        elif len(answer) > self.max_length:
            issues.append("Too long")
        for pattern in self.noise_patterns:
            if re.search(pattern, answer):
                issues.append("Contains metadata/noise")
                break
        if question:
            qw = set(question.lower().split())
            aw = answer.lower().split()
            if len(aw) > 0 and len(set(aw) & qw) > len(aw) * 0.7:
                issues.append("Mostly repeats question")
        if len(answer.split()) < 3:
            issues.append("Too few words")
        return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# QA Generator
# ---------------------------------------------------------------------------

class EnhancedQAPairGenerator:
    """Two-pass QA pair generator.

    Only T5-family models (0=T5-QA-QG, 1=FLAN-T5) are supported.
    """

    def __init__(self, model_choice=0, weights_dir="weights", use_extractive=True):
        self.weights_dir = weights_dir
        self.model_choice = model_choice
        self.use_extractive = use_extractive

        self.text_processor = TextProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.question_validator = QuestionValidator()
        self.answer_validator = AnswerValidator()

        self.model_configs = {
            0: {"name": "valhalla/t5-small-qa-qg-hl", "type": "t5"},
            1: {"name": "google/flan-t5-base", "type": "flan-t5"},
        }
        if model_choice not in self.model_configs:
            raise ValueError(f"Invalid model choice {model_choice}. Use 0 or 1.")

        self.qa_pipeline = self._load_qa_model()
        self.extractive_pipeline = self._load_extractive_model() if use_extractive else None

    # -- model loading -------------------------------------------------------

    def _load_qa_model(self):
        os.makedirs(self.weights_dir, exist_ok=True)
        config = self.model_configs[self.model_choice]
        logger.info("Loading %s model: %s", config["type"].upper(), config["name"])
        model = T5ForConditionalGeneration.from_pretrained(config["name"], cache_dir=self.weights_dir)
        tokenizer = T5Tokenizer.from_pretrained(config["name"], cache_dir=self.weights_dir)
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

    def _load_extractive_model(self):
        logger.info("Loading extractive QA model...")
        try:
            name = "distilbert-base-uncased-distilled-squad"
            model = AutoModelForQuestionAnswering.from_pretrained(name, cache_dir=self.weights_dir)
            tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=self.weights_dir)
            device = 0 if torch.cuda.is_available() else -1
            return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
        except Exception as e:
            logger.error("Error loading extractive model: %s", e)
            return None

    # -- PDF & chunking ------------------------------------------------------

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total = len(pdf.pages)
                logger.info("Processing all %d pages from PDF...", total)
                for i, page in enumerate(pdf.pages):
                    if (i + 1) % 10 == 0:
                        logger.info("Processed %d/%d pages...", i + 1, total)
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception as e:
            logger.error("Error extracting text from PDF: %s", e)
            return ""
        return self.text_processor.clean_extracted_text(text)

    def create_smart_chunks(self, text: str, chunk_size: int = 4) -> List[Dict]:
        sentences = self._split_into_sentences(text)
        chunks: List[Dict] = []
        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i : i + chunk_size]
            chunk_text = " ".join(chunk_sentences)
            if len(chunk_text.strip()) > 100:
                content_type = self.content_analyzer.identify_content_type(chunk_text)
                key_concepts = self.content_analyzer.extract_key_concepts(chunk_text)
                chunks.append({
                    "text": self.text_processor.restore_math_sequences(chunk_text.strip()),
                    "content_type": content_type,
                    "key_concepts": key_concepts,
                    "sentence_range": (i, min(i + chunk_size, len(sentences))),
                })
            i += max(1, chunk_size - 1)
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])", text)
        cleaned: List[str] = []
        for s in sentences:
            s = s.strip()
            if (len(s) > 20 and not re.match(r"^\d+$", s)
                    and not re.match(r"^[A-Z\s]{3,}$", s)
                    and not re.search(r"\d{2}:\d{2}:\d{2}", s)):
                cleaned.append(s)
        return cleaned

    # -- question generation -------------------------------------------------

    def generate_contextual_questions(self, chunk: Dict) -> List[str]:
        context = chunk["text"]
        content_type = chunk["content_type"]
        prompts = {
            "definition": f"Generate clear questions asking for definitions from: {context}",
            "sequence": f"Create questions about mathematical patterns and sequences in: {context}",
            "explanation": f"Form questions asking why or how things work in: {context}",
            "example": f"Generate questions about examples and applications in: {context}",
            "instruction": f"Create questions that would help understand the instructions in: {context}",
            "mathematical": f"Generate math questions based on: {context}",
        }
        prompt = prompts.get(content_type, f"Generate educational questions from: {context}")
        try:
            result = self.qa_pipeline(
                prompt, max_length=150, min_length=20, num_return_sequences=2,
                temperature=0.4, top_p=0.9, repetition_penalty=1.2, do_sample=True,
                pad_token_id=getattr(self.qa_pipeline.tokenizer, "eos_token_id", 0),
            )
            questions: List[str] = []
            items = result if isinstance(result, list) else [result]
            for res in items:
                questions.extend(self._extract_questions_from_text(res.get("generated_text", "").strip()))
            return [q for q in questions if self.question_validator.is_valid_question(q)[0]][:3]
        except Exception as e:
            logger.error("Error generating questions: %s", e)
            return []

    def _extract_questions_from_text(self, text: str) -> List[str]:
        questions: List[str] = []
        for q in re.split(r"\?+", text):
            q = q.strip()
            if q and len(q) > 10:
                q = re.sub(r"^(questions?:?\s*)", "", q, flags=re.IGNORECASE)
                q = re.sub(r"^(\d+\.?\s*)", "", q).strip()
                if q and not q.endswith("?"):
                    q += "?"
                if len(q) > 15:
                    questions.append(q)
        return questions

    # -- answer generation ---------------------------------------------------

    def generate_best_answer(self, question: str, chunk: Dict) -> Tuple[str, str]:
        context = chunk["text"]
        answers: List[str] = []
        sources: List[str] = []

        if self.extractive_pipeline:
            try:
                result = self.extractive_pipeline(question=question, context=context)
                answer = result.get("answer", "").strip()
                if result.get("score", 0) > 0.3 and len(answer) > 5:
                    if self.answer_validator.is_valid_answer(answer, question)[0]:
                        answers.append(answer)
                        sources.append("extractive")
            except Exception:
                pass

        try:
            config = self.model_configs[self.model_choice]
            if config["type"] == "flan-t5":
                input_text = f"Answer this question based on the context: {question}\nContext: {context}"
            else:
                input_text = f"answer: {question} context: {context}"
            result = self.qa_pipeline(
                input_text, max_length=100, min_length=10, temperature=0.3,
                do_sample=True, repetition_penalty=1.1,
                pad_token_id=getattr(self.qa_pipeline.tokenizer, "eos_token_id", 0),
            )
            raw = result[0].get("generated_text", "").strip() if isinstance(result, list) else result.get("generated_text", "").strip()
            answer = self._clean_answer(raw, question)
            if answer and self.answer_validator.is_valid_answer(answer, question)[0]:
                answers.append(answer)
                sources.append("generative")
        except Exception:
            pass

        if answers:
            if "extractive" in sources:
                idx = sources.index("extractive")
                return answers[idx], sources[idx]
            return answers[0], sources[0]
        return "", "none"

    def _clean_answer(self, answer: str, question: str) -> str:
        answer = re.sub(r"^(answer:?\s*)", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"^(the answer is:?\s*)", "", answer, flags=re.IGNORECASE)
        qw = set(question.lower().split())
        aw = answer.lower().split()
        if len(aw) > 0 and len(set(aw) & qw) > len(aw) * 0.7:
            return ""
        return answer.strip()

    # -- extractive verification ---------------------------------------------

    @staticmethod
    def _verify_extractive(pair: Dict) -> bool:
        """Return True only if the answer appears verbatim in the context.

        This guarantees the pair is compatible with extractive QA training
        (DistilBERT), where token-level start/end positions must be found.
        """
        answer = pair.get("answer", "").strip()
        context = pair.get("context", "")
        return answer != "" and answer in context

    # -- deduplication -------------------------------------------------------

    def _deduplicate_pairs(self, qa_pairs: List[Dict], threshold: float = 0.75) -> List[Dict]:
        if not qa_pairs:
            return qa_pairs
        unique: List[Dict] = []
        seen: List[str] = []
        for pair in qa_pairs:
            q = pair["question"].lower()
            if not any(SequenceMatcher(None, q, s).ratio() > threshold for s in seen):
                unique.append(pair)
                seen.append(q)
        removed = len(qa_pairs) - len(unique)
        if removed:
            logger.info("Deduplication removed %d near-duplicate pairs", removed)
        return unique

    # -- pass 1 --------------------------------------------------------------

    def generate_qa_pairs_pass1(self, text: str) -> List[Dict]:
        logger.info("Pass 1: Enhanced QA pair generation...")
        chunks = self.create_smart_chunks(text, chunk_size=5)
        if len(chunks) < 2:
            logger.warning("Not enough content chunks for meaningful QA pairs")
            return []
        logger.info("Created %d content-aware chunks", len(chunks))
        qa_pairs: List[Dict] = []
        for i, chunk in enumerate(chunks):
            logger.info("Processing chunk %d/%d (Type: %s)", i + 1, len(chunks), chunk["content_type"])
            for question in self.generate_contextual_questions(chunk):
                answer, source = self.generate_best_answer(question, chunk)
                if answer:
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "context": chunk["text"],
                        "source": source,
                        "content_type": chunk["content_type"],
                        "key_concepts": chunk["key_concepts"],
                        "model_used": self.model_configs[self.model_choice]["name"],
                        "quality_score": self._calculate_quality_score(question, answer, chunk["text"]),
                    })

        qa_pairs = self._deduplicate_pairs(qa_pairs)

        # Enforce extractive compatibility: answer must be a substring of context
        before = len(qa_pairs)
        qa_pairs = [p for p in qa_pairs if self._verify_extractive(p)]
        dropped = before - len(qa_pairs)
        if dropped:
            logger.info(
                "Extractive filter removed %d pairs whose answer was not "
                "a verbatim substring of the context", dropped,
            )

        logger.info("Generated %d QA pairs in Pass 1 (after dedup + extractive filter)", len(qa_pairs))
        return qa_pairs

    def _calculate_quality_score(self, question: str, answer: str, context: str) -> float:
        score = 0.0
        if self.question_validator.is_valid_question(question)[0]:
            score += 0.3
        if self.answer_validator.is_valid_answer(answer, question)[0]:
            score += 0.3
        qw = set(question.lower().split())
        aw = set(answer.lower().split())
        cw = set(context.lower().split())
        score += min(0.2, len(qw & aw) / max(len(qw), 1) * 0.4)
        score += min(0.2, len(aw & cw) / max(len(aw), 1) * 0.4)
        return min(1.0, score)


# ---------------------------------------------------------------------------
# QA Enhancer (Pass 2)
# ---------------------------------------------------------------------------

class QAEnhancer:
    def __init__(self):
        self.question_validator = QuestionValidator()
        self.answer_validator = AnswerValidator()
        self.text_processor = TextProcessor()

    def enhance_qa_pairs(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], Dict]:
        logger.info("Pass 2: Enhancing QA pairs...")
        enhanced: List[Dict] = []
        stats: Dict = {"filtered_out": 0, "enhanced": 0, "quality_scores": []}
        for pair in qa_pairs:
            ep = self._enhance_single_pair(pair)
            if ep:
                enhanced.append(ep)
                stats["enhanced"] += 1
                stats["quality_scores"].append(ep.get("quality_score", 0))
            else:
                stats["filtered_out"] += 1
        logger.info("Enhanced: %d, Filtered out: %d", stats["enhanced"], stats["filtered_out"])
        return enhanced, stats

    def _enhance_single_pair(self, pair: Dict) -> Optional[Dict]:
        q = self.text_processor.fix_encoding(pair.get("question", "").strip())
        a = self.text_processor.fix_encoding(pair.get("answer", "").strip())
        c = self.text_processor.fix_encoding(pair.get("context", "").strip())
        if not self.question_validator.is_valid_question(q)[0]:
            return None
        if not self.answer_validator.is_valid_answer(a, q)[0]:
            return None
        score = self._score(q, a, c)
        if score < 0.4:
            return None
        return {
            **pair, "question": q, "answer": a, "context": c,
            "quality_score": score,
            "question_length": len(q), "answer_length": len(a),
            "question_type": self._classify_question_type(q),
            "difficulty_level": self._assess_difficulty(q, a),
            "topic_keywords": self._extract_topic_keywords(c),
        }

    def _score(self, q: str, a: str, c: str) -> float:
        s = 0.0
        if 20 <= len(q) <= 150: s += 0.2
        if 10 <= len(a) <= 200: s += 0.2
        if q.count("?") == 1 and q.endswith("?"): s += 0.1
        aw = a.lower().split(); cw = c.lower().split(); qw = q.lower().split()
        ac = len(set(aw) & set(cw)) / max(len(aw), 1)
        if 0.2 <= ac <= 0.8: s += 0.2
        qa = len(set(qw) & set(aw)) / max(len(qw), 1)
        if 0.1 <= qa <= 0.5: s += 0.15
        if len(set(aw)) / max(len(aw), 1) > 0.7: s += 0.15
        return min(1.0, s)

    def _classify_question_type(self, q: str) -> str:
        ql = q.lower()
        if ql.startswith(("what is", "what are", "what does")): return "definition"
        if ql.startswith(("how",)): return "explanation"
        if ql.startswith(("why",)): return "reasoning"
        if any(w in ql for w in ("calculate", "solve", "find the value")): return "computation"
        if "sequence" in ql or "pattern" in ql: return "mathematical_pattern"
        if "example" in ql or "such as" in ql: return "example"
        return "general"

    def _assess_difficulty(self, q: str, a: str) -> str:
        qw = len(q.split()); aw = len(a.split())
        combined = q.lower() + a.lower()
        complex_terms = ["theorem", "proof", "algorithm", "derivative", "integral",
                         "matrix", "quadratic", "polynomial", "logarithm"]
        if any(t in combined for t in complex_terms): return "advanced"
        if any(t in q.lower() for t in ["and", "relationship", "compare", "analyze", "explain why"]): return "advanced"
        if qw > 20 and aw > 30: return "advanced"
        if qw > 15 or aw > 20: return "intermediate"
        return "basic"

    def _extract_topic_keywords(self, context: str) -> List[str]:
        patterns = [
            r"\b(?:pattern|sequence|number|triangle|square|polygon|geometry|"
            r"mathematics|equation|fraction|ratio|percent|angle|perimeter|area|volume)\b",
            r"\b(?:definition|example|explanation|theory|concept|principle)\b",
            r"\b(?:calculate|solve|find|determine|identify|analyze)\b",
        ]
        kw: List[str] = []
        for p in patterns:
            kw.extend(re.findall(p, context.lower()))
        return list(set(kw))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-pass QA pair generator with enhancement")
    parser.add_argument("-i", "--input", required=True, help="Path to input PDF file")
    parser.add_argument("-o", "--output", required=True, help="Path for output JSON file")
    parser.add_argument("-w", "--weights", default="weights", help="Directory to store model weights")
    parser.add_argument("-m", "--model", type=int, choices=[0, 1], default=0,
                        help="Model choice: 0=T5-QA-QG (default), 1=FLAN-T5")
    parser.add_argument("-x", "--extractive", action="store_true",
                        help="Enable extractive QA model for better answer generation")
    parser.add_argument("--enhance-only", help="Path to existing JSON file to enhance (skip Pass 1)")
    parser.add_argument("--export-chunks", help="Path to save chunk index JSON (for retrieval in querySLM)")

    args = parser.parse_args()
    model_names = {0: "valhalla/t5-small-qa-qg-hl", 1: "google/flan-t5-base"}

    logger.info("Selected model: %s", model_names[args.model])
    logger.info("Extractive QA: %s", "ENABLED" if args.extractive else "DISABLED")

    if args.enhance_only:
        logger.info("Enhancement-only mode: processing %s", args.enhance_only)
        with open(args.enhance_only, "r", encoding="utf-8") as f:
            data = json.load(f)
        qa_pairs = data.get("qa_pairs", [])
        logger.info("Loaded %d existing QA pairs", len(qa_pairs))

        enhancer = QAEnhancer()
        enhanced_pairs, quality_stats = enhancer.enhance_qa_pairs(qa_pairs)
        avg = sum(quality_stats["quality_scores"]) / max(len(quality_stats["quality_scores"]), 1)

        out = {
            "qa_pairs": enhanced_pairs, "total_pairs": len(enhanced_pairs),
            "metadata": {**data.get("metadata", {}), "enhancement_applied": True,
                         "original_count": len(qa_pairs), "filtered_count": quality_stats["filtered_out"]},
            "quality_stats": {**data.get("quality_stats", {}), "average_quality_score": avg,
                              "enhancement_stats": quality_stats},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)
        logger.info("Enhanced QA pairs saved to %s", args.output)
    else:
        try:
            generator = EnhancedQAPairGenerator(
                model_choice=args.model, weights_dir=args.weights, use_extractive=args.extractive,
            )
            logger.info("Models loaded successfully for Pass 1")
            text = generator.extract_text_from_pdf(args.input)
            if not text:
                logger.error("No text extracted from PDF."); return
            logger.info("Extracted %d characters from PDF", len(text))

            # Export chunk index if requested
            if args.export_chunks:
                chunks = generator.create_smart_chunks(text, chunk_size=5)
                chunk_index = [{"id": i, "text": c["text"], "content_type": c["content_type"],
                                "key_concepts": c["key_concepts"]} for i, c in enumerate(chunks)]
                with open(args.export_chunks, "w", encoding="utf-8") as f:
                    json.dump({"chunks": chunk_index, "total": len(chunk_index)}, f, indent=2, ensure_ascii=False)
                logger.info("Exported %d chunks to %s", len(chunk_index), args.export_chunks)

            qa_pairs = generator.generate_qa_pairs_pass1(text)
            if not qa_pairs:
                logger.error("No QA pairs generated in Pass 1."); return
            logger.info("Pass 1 completed: %d QA pairs generated", len(qa_pairs))

            del generator
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            enhancer = QAEnhancer()
            enhanced_pairs, quality_stats = enhancer.enhance_qa_pairs(qa_pairs)
            avg = sum(quality_stats["quality_scores"]) / max(len(quality_stats["quality_scores"]), 1)

            final = {
                "qa_pairs": enhanced_pairs, "total_pairs": len(enhanced_pairs),
                "metadata": {"model_used": model_names[args.model], "extractive_enabled": args.extractive,
                             "passes_completed": 2, "source_file": os.path.basename(args.input),
                             "original_pass1_count": len(qa_pairs), "final_enhanced_count": len(enhanced_pairs),
                             "filtered_count": quality_stats["filtered_out"]},
                "quality_stats": {"average_quality_score": avg, "total_generated": len(qa_pairs),
                                  "total_enhanced": quality_stats["enhanced"],
                                  "total_filtered": quality_stats["filtered_out"],
                                  "enhancement_stats": quality_stats},
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(final, f, indent=4, ensure_ascii=False)
            logger.info("Final QA pairs saved to %s", args.output)

            logger.info("=" * 60)
            logger.info("PROCESSING COMPLETE")
            logger.info("Source: %s | Model: %s | Extractive: %s",
                        os.path.basename(args.input), model_names[args.model],
                        "Yes" if args.extractive else "No")
            logger.info("Chars: %d | Pass1: %d | Enhanced: %d | Filtered: %d | Final: %d | Avg score: %.3f",
                        len(text), len(qa_pairs), quality_stats["enhanced"],
                        quality_stats["filtered_out"], len(enhanced_pairs), avg)
            logger.info("=" * 60)

        except Exception as e:
            logger.error("Error during processing: %s", e)
            raise


if __name__ == "__main__":
    main()