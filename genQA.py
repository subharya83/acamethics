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

# Setup logging (replaces print() throughout)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        """Fix common encoding issues."""
        for bad, good in self.encoding_fixes.items():
            text = text.replace(bad, good)
        return text

    def remove_noise(self, text: str) -> str:
        """Remove page headers, footers, and other noise."""
        for pattern in self.noise_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()

    def preserve_math_sequences(self, text: str) -> str:
        """Identify and preserve mathematical sequences with indexed markers."""
        seq_pattern = r"(\d+(?:,\s*\d+){3,}(?:,\s*\.{3})?)"
        sequences = re.findall(seq_pattern, text)

        # Use indexed markers to avoid ambiguity on restore
        for idx, seq in enumerate(sequences):
            text = text.replace(seq, f"[MATH_SEQ_{idx}:{seq}]", 1)

        return text

    def restore_math_sequences(self, text: str) -> str:
        """Restore mathematical sequences from indexed markers."""
        return re.sub(r"\[MATH_SEQ_\d+:(.*?)\]", r"\1", text)

    def clean_extracted_text(self, text: str) -> str:
        """Complete text cleaning pipeline."""
        text = self.fix_encoding(text)
        text = self.remove_noise(text)
        text = self.preserve_math_sequences(text)
        return text


class ContentAnalyzer:
    """Analyzes content type and extracts key information."""

    def __init__(self):
        # Separate plain-text indicators from regex patterns to avoid the
        # buggy ``indicator.startswith('r')`` check in the original code.
        self.text_indicators: Dict[str, List[str]] = {
            "definition": [
                "is defined as", "refers to", "means", "what is", "called",
            ],
            "example": ["for example", "such as", "including", "like", "instance"],
            "sequence": ["sequence", "pattern", "next", "series"],
            "explanation": [
                "because", "due to", "reason", "why", "how", "explains",
            ],
            "instruction": ["draw", "copy", "find", "calculate", "solve", "can you"],
            "mathematical": ["numbers", "triangular", "square", "cube"],
        }

        # Regex patterns stored separately as compiled objects
        self.regex_indicators: Dict[str, List[re.Pattern]] = {
            "mathematical": [re.compile(r"\d+(?:,\s*\d+){2,}")],
        }

    def identify_content_type(self, text: str) -> str:
        """Identify the primary content type of a text chunk."""
        text_lower = text.lower()
        scores: Dict[str, int] = {}

        for content_type, indicators in self.text_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in text_lower:
                    score += 1
            scores[content_type] = score

        for content_type, patterns in self.regex_indicators.items():
            for pat in patterns:
                if pat.search(text_lower):
                    scores[content_type] = scores.get(content_type, 0) + 2

        return max(scores, key=scores.get) if scores else "general"

    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key mathematical and educational concepts."""
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
        concepts.extend(
            [term.lower() for term in cap_terms if len(term.split()) <= 3]
        )

        return list(set(concepts))


class QuestionValidator:
    """Validates question quality and structure."""

    def __init__(self):
        self.min_length = 15
        self.max_length = 200
        self.question_starters = [
            "what", "how", "why", "when", "where", "which", "who",
            "can", "do", "does", "is", "are", "if", "find", "calculate", "solve",
        ]

    def is_valid_question(self, question: str) -> Tuple[bool, List[str]]:
        """Validate question structure and content."""
        issues: List[str] = []

        if not question or len(question.strip()) == 0:
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
            starter in question.lower()[:20] for starter in self.question_starters
        ):
            issues.append("Doesn't start like a question")

        if question.count(".") >= question.count("?"):
            issues.append("Contains statement fragments")

        words = question.lower().split()
        if len(words) > 3 and len(set(words)) < len(words) * 0.6:
            issues.append("Too repetitive")

        return len(issues) == 0, issues


class AnswerValidator:
    """Validates answer quality and relevance."""

    def __init__(self):
        self.min_length = 5
        self.max_length = 300
        self.noise_patterns = [
            r"\d{2}:\d{2}:\d{2}",
            r"Chapter \d+",
            r"\.indd",
            r"^[A-Z\s]{5,}$",
            r"^\d+$",
            r"^[^a-zA-Z0-9]*$",
        ]

    def is_valid_answer(
        self, answer: str, question: str = ""
    ) -> Tuple[bool, List[str]]:
        """Validate answer quality and relevance."""
        issues: List[str] = []

        if not answer or len(answer.strip()) == 0:
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
            question_words = set(question.lower().split())
            answer_words = answer.lower().split()
            overlap = len(set(answer_words) & question_words)
            if len(answer_words) > 0 and overlap > len(answer_words) * 0.7:
                issues.append("Mostly repeats question")

        if len(answer.split()) < 3:
            issues.append("Too few words")

        return len(issues) == 0, issues


class EnhancedQAPairGenerator:
    """Enhanced QA pair generator with better extraction.

    Only T5-based models (options 0 and 1) are supported for question
    generation. The original BART-CNN option was a summarization model
    that could not generate questions and has been removed.
    """

    def __init__(
        self,
        model_choice: int = 0,
        weights_dir: str = "weights",
        use_extractive: bool = True,
    ):
        self.weights_dir = weights_dir
        self.model_choice = model_choice
        self.use_extractive = use_extractive

        self.text_processor = TextProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.question_validator = QuestionValidator()
        self.answer_validator = AnswerValidator()

        # Only T5-family models that can actually generate questions
        self.model_configs = {
            0: {"name": "valhalla/t5-small-qa-qg-hl", "type": "t5"},
            1: {"name": "google/flan-t5-base", "type": "flan-t5"},
        }

        if model_choice not in self.model_configs:
            raise ValueError(
                f"Invalid model choice {model_choice}. Use 0 (T5-QA-QG) or 1 (FLAN-T5)."
            )

        self.qa_pipeline = self._load_qa_model()
        self.extractive_pipeline = (
            self._load_extractive_model() if use_extractive else None
        )

    def _load_qa_model(self):
        """Load QA generation model."""
        os.makedirs(self.weights_dir, exist_ok=True)
        config = self.model_configs[self.model_choice]

        logger.info("Loading %s model: %s", config["type"].upper(), config["name"])

        try:
            model = T5ForConditionalGeneration.from_pretrained(
                config["name"], cache_dir=self.weights_dir
            )
            tokenizer = T5Tokenizer.from_pretrained(
                config["name"], cache_dir=self.weights_dir
            )

            device = 0 if torch.cuda.is_available() else -1
            return pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def _load_extractive_model(self):
        """Load extractive QA model."""
        logger.info("Loading extractive QA model...")
        try:
            model_name = "distilbert-base-uncased-distilled-squad"
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_name, cache_dir=self.weights_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=self.weights_dir
            )

            device = 0 if torch.cuda.is_available() else -1
            return pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        except Exception as e:
            logger.error("Error loading extractive model: %s", e)
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract and clean text from entire PDF."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info("Processing all %d pages from PDF...", total_pages)

                for i, page in enumerate(pdf.pages):
                    if (i + 1) % 10 == 0:
                        logger.info("Processed %d/%d pages...", i + 1, total_pages)

                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception as e:
            logger.error("Error extracting text from PDF: %s", e)
            return ""

        return self.text_processor.clean_extracted_text(text)

    def create_smart_chunks(self, text: str, chunk_size: int = 4) -> List[Dict]:
        """Create context chunks with content analysis."""
        sentences = self._split_into_sentences(text)
        chunks: List[Dict] = []

        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i : i + chunk_size]
            chunk_text = " ".join(chunk_sentences)

            if len(chunk_text.strip()) > 100:
                content_type = self.content_analyzer.identify_content_type(chunk_text)
                key_concepts = self.content_analyzer.extract_key_concepts(chunk_text)

                chunks.append(
                    {
                        "text": self.text_processor.restore_math_sequences(
                            chunk_text.strip()
                        ),
                        "content_type": content_type,
                        "key_concepts": key_concepts,
                        "sentence_range": (i, min(i + chunk_size, len(sentences))),
                    }
                )

            i += max(1, chunk_size - 1)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Improved sentence splitting."""
        sentences = re.split(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])", text
        )

        cleaned: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (
                len(sentence) > 20
                and not re.match(r"^\d+$", sentence)
                and not re.match(r"^[A-Z\s]{3,}$", sentence)
                and not re.search(r"\d{2}:\d{2}:\d{2}", sentence)
            ):
                cleaned.append(sentence)

        return cleaned

    def generate_contextual_questions(self, chunk: Dict) -> List[str]:
        """Generate questions based on content type."""
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

        prompt = prompts.get(
            content_type, f"Generate educational questions from: {context}"
        )

        try:
            result = self.qa_pipeline(
                prompt,
                max_length=150,
                min_length=20,
                num_return_sequences=2,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=getattr(
                    self.qa_pipeline.tokenizer, "eos_token_id", 0
                ),
            )

            questions: List[str] = []
            if isinstance(result, list):
                for res in result:
                    generated_text = res.get("generated_text", "").strip()
                    questions.extend(
                        self._extract_questions_from_text(generated_text)
                    )
            else:
                generated_text = result.get("generated_text", "").strip()
                questions.extend(self._extract_questions_from_text(generated_text))

            valid_questions: List[str] = []
            for q in questions:
                is_valid, _ = self.question_validator.is_valid_question(q)
                if is_valid:
                    valid_questions.append(q)

            return valid_questions[:3]

        except Exception as e:
            logger.error("Error generating questions: %s", e)
            return []

    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Extract individual questions from generated text."""
        questions: List[str] = []
        potential_questions = re.split(r"\?+", text)

        for q in potential_questions:
            q = q.strip()
            if q and len(q) > 10:
                q = re.sub(r"^(questions?:?\s*)", "", q, flags=re.IGNORECASE)
                q = re.sub(r"^(\d+\.?\s*)", "", q)
                q = q.strip()

                if q and not q.endswith("?"):
                    q += "?"

                if len(q) > 15:
                    questions.append(q)

        return questions

    def generate_best_answer(self, question: str, chunk: Dict) -> Tuple[str, str]:
        """Generate the best possible answer using multiple methods."""
        context = chunk["text"]
        answers: List[str] = []
        sources: List[str] = []

        # Try extractive QA first (usually most reliable)
        if self.extractive_pipeline:
            try:
                result = self.extractive_pipeline(question=question, context=context)
                answer = result.get("answer", "").strip()
                confidence = result.get("score", 0)

                if confidence > 0.3 and len(answer) > 5:
                    is_valid, _ = self.answer_validator.is_valid_answer(answer, question)
                    if is_valid:
                        answers.append(answer)
                        sources.append("extractive")
            except Exception:
                pass

        # Try generative model
        try:
            config = self.model_configs[self.model_choice]

            if config["type"] == "t5":
                input_text = f"answer: {question} context: {context}"
            elif config["type"] == "flan-t5":
                input_text = (
                    f"Answer this question based on the context: {question}\n"
                    f"Context: {context}"
                )
            else:
                input_text = f"answer: {question} context: {context}"

            result = self.qa_pipeline(
                input_text,
                max_length=100,
                min_length=10,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=getattr(
                    self.qa_pipeline.tokenizer, "eos_token_id", 0
                ),
            )

            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get("generated_text", "").strip()
            else:
                answer = result.get("generated_text", "").strip()

            answer = self._clean_answer(answer, question)

            if answer:
                is_valid, _ = self.answer_validator.is_valid_answer(answer, question)
                if is_valid:
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
        """Clean and validate generated answers."""
        answer = re.sub(r"^(answer:?\s*)", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"^(the answer is:?\s*)", "", answer, flags=re.IGNORECASE)

        question_words = set(question.lower().split())
        answer_words = answer.lower().split()

        if len(answer_words) > 0:
            overlap = len(set(answer_words) & question_words)
            if overlap > len(answer_words) * 0.7:
                return ""

        return answer.strip()

    def _deduplicate_pairs(
        self, qa_pairs: List[Dict], threshold: float = 0.75
    ) -> List[Dict]:
        """Remove near-duplicate QA pairs based on question similarity."""
        if not qa_pairs:
            return qa_pairs

        unique_pairs: List[Dict] = []
        seen_questions: List[str] = []

        for pair in qa_pairs:
            q = pair["question"].lower()
            is_dup = False
            for seen_q in seen_questions:
                ratio = SequenceMatcher(None, q, seen_q).ratio()
                if ratio > threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique_pairs.append(pair)
                seen_questions.append(q)

        removed = len(qa_pairs) - len(unique_pairs)
        if removed > 0:
            logger.info("Deduplication removed %d near-duplicate pairs", removed)

        return unique_pairs

    def generate_qa_pairs_pass1(self, text: str) -> List[Dict[str, str]]:
        """First pass: Generate QA pairs with enhanced extraction."""
        logger.info("Pass 1: Enhanced QA pair generation...")

        chunks = self.create_smart_chunks(text, chunk_size=5)
        if len(chunks) < 2:
            logger.warning("Not enough content chunks for meaningful QA pairs")
            return []

        logger.info("Created %d content-aware chunks", len(chunks))
        qa_pairs: List[Dict] = []

        for i, chunk in enumerate(chunks):
            logger.info(
                "Processing chunk %d/%d (Type: %s)",
                i + 1, len(chunks), chunk["content_type"],
            )

            questions = self.generate_contextual_questions(chunk)

            for question in questions:
                answer, source = self.generate_best_answer(question, chunk)

                if answer:
                    qa_pair = {
                        "question": question,
                        "answer": answer,
                        "context": chunk["text"],
                        "source": source,
                        "content_type": chunk["content_type"],
                        "key_concepts": chunk["key_concepts"],
                        "model_used": self.model_configs[self.model_choice]["name"],
                        "quality_score": self._calculate_quality_score(
                            question, answer, chunk["text"]
                        ),
                    }
                    qa_pairs.append(qa_pair)

        # Deduplicate before returning
        qa_pairs = self._deduplicate_pairs(qa_pairs)
        logger.info("Generated %d QA pairs in Pass 1 (after dedup)", len(qa_pairs))
        return qa_pairs

    def _calculate_quality_score(
        self, question: str, answer: str, context: str
    ) -> float:
        """Calculate quality score for QA pair."""
        score = 0.0

        is_valid_q, _ = self.question_validator.is_valid_question(question)
        if is_valid_q:
            score += 0.3

        is_valid_a, _ = self.answer_validator.is_valid_answer(answer, question)
        if is_valid_a:
            score += 0.3

        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        c_words = set(context.lower().split())

        qa_overlap = len(q_words & a_words) / max(len(q_words), 1)
        ac_overlap = len(a_words & c_words) / max(len(a_words), 1)

        score += min(0.2, qa_overlap * 0.4)
        score += min(0.2, ac_overlap * 0.4)

        return min(1.0, score)


class QAEnhancer:
    """Second pass: Enhance existing QA pairs JSON."""

    def __init__(self):
        self.question_validator = QuestionValidator()
        self.answer_validator = AnswerValidator()
        self.text_processor = TextProcessor()

    def enhance_qa_pairs(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Enhance QA pairs list and return stats."""
        logger.info("Pass 2: Enhancing QA pairs...")

        enhanced_pairs: List[Dict] = []
        quality_stats: Dict = {
            "filtered_out": 0,
            "enhanced": 0,
            "quality_scores": [],
        }

        for pair in qa_pairs:
            enhanced_pair = self._enhance_single_pair(pair)

            if enhanced_pair:
                enhanced_pairs.append(enhanced_pair)
                quality_stats["enhanced"] += 1
                quality_stats["quality_scores"].append(
                    enhanced_pair.get("quality_score", 0)
                )
            else:
                quality_stats["filtered_out"] += 1

        logger.info(
            "Enhanced: %d, Filtered out: %d",
            quality_stats["enhanced"], quality_stats["filtered_out"],
        )
        return enhanced_pairs, quality_stats

    def _enhance_single_pair(self, pair: Dict) -> Optional[Dict]:
        """Enhance a single QA pair."""
        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()
        context = pair.get("context", "").strip()

        question = self.text_processor.fix_encoding(question)
        answer = self.text_processor.fix_encoding(answer)
        context = self.text_processor.fix_encoding(context)

        is_valid_q, _ = self.question_validator.is_valid_question(question)
        if not is_valid_q:
            return None

        is_valid_a, _ = self.answer_validator.is_valid_answer(answer, question)
        if not is_valid_a:
            return None

        quality_score = self._calculate_enhanced_quality_score(question, answer, context)

        if quality_score < 0.4:
            return None

        return {
            **pair,
            "question": question,
            "answer": answer,
            "context": context,
            "quality_score": quality_score,
            "question_length": len(question),
            "answer_length": len(answer),
            "question_type": self._classify_question_type(question),
            "difficulty_level": self._assess_difficulty(question, answer),
            "topic_keywords": self._extract_topic_keywords(context),
        }

    def _calculate_enhanced_quality_score(
        self, question: str, answer: str, context: str
    ) -> float:
        """Enhanced quality scoring."""
        score = 0.0

        if 20 <= len(question) <= 150:
            score += 0.2
        if 10 <= len(answer) <= 200:
            score += 0.2

        if question.count("?") == 1 and question.endswith("?"):
            score += 0.1

        q_words = question.lower().split()
        a_words = answer.lower().split()
        c_words = context.lower().split()

        ac_overlap = len(set(a_words) & set(c_words)) / max(len(a_words), 1)
        if 0.2 <= ac_overlap <= 0.8:
            score += 0.2

        qa_overlap = len(set(q_words) & set(a_words)) / max(len(q_words), 1)
        if 0.1 <= qa_overlap <= 0.5:
            score += 0.15

        unique_ratio = len(set(a_words)) / max(len(a_words), 1)
        if unique_ratio > 0.7:
            score += 0.15

        return min(1.0, score)

    def _classify_question_type(self, question: str) -> str:
        """Classify question type."""
        q_lower = question.lower()

        if q_lower.startswith(("what is", "what are", "what does")):
            return "definition"
        elif q_lower.startswith(("how", "how can", "how do")):
            return "explanation"
        elif q_lower.startswith(("why", "why do", "why does")):
            return "reasoning"
        elif any(w in q_lower for w in ("calculate", "solve", "find the value")):
            return "computation"
        elif "sequence" in q_lower or "pattern" in q_lower:
            return "mathematical_pattern"
        elif "example" in q_lower or "such as" in q_lower:
            return "example"
        else:
            return "general"

    def _assess_difficulty(self, question: str, answer: str) -> str:
        """Assess difficulty level."""
        q_words = len(question.split())
        a_words = len(answer.split())

        complex_terms = [
            "theorem", "proof", "algorithm", "derivative", "integral",
            "matrix", "quadratic", "polynomial", "logarithm",
        ]
        has_complex = any(
            term in question.lower() + answer.lower() for term in complex_terms
        )

        concept_indicators = ["and", "relationship", "compare", "analyze", "explain why"]
        multi_concept = any(ind in question.lower() for ind in concept_indicators)

        if has_complex or multi_concept or (q_words > 20 and a_words > 30):
            return "advanced"
        elif q_words > 15 or a_words > 20:
            return "intermediate"
        else:
            return "basic"

    def _extract_topic_keywords(self, context: str) -> List[str]:
        """Extract topic-relevant keywords."""
        topic_patterns = [
            r"\b(?:pattern|sequence|number|triangle|square|polygon|"
            r"geometry|mathematics|equation|fraction|ratio|percent|"
            r"angle|perimeter|area|volume)\b",
            r"\b(?:definition|example|explanation|theory|concept|principle)\b",
            r"\b(?:calculate|solve|find|determine|identify|analyze)\b",
        ]

        keywords: List[str] = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, context.lower())
            keywords.extend(matches)

        return list(set(keywords))


def main():
    """Main function to run both passes and generate single output file."""
    parser = argparse.ArgumentParser(
        description="Two-pass QA pair generator with enhancement"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input PDF file")
    parser.add_argument("-o", "--output", required=True, help="Path for output JSON file")
    parser.add_argument(
        "-w", "--weights", default="weights", help="Directory to store model weights"
    )
    parser.add_argument(
        "-m", "--model", type=int, choices=[0, 1], default=0,
        help="Model choice: 0=T5-QA-QG (default), 1=FLAN-T5",
    )
    parser.add_argument(
        "-x", "--extractive", action="store_true",
        help="Enable extractive QA model for better answer generation",
    )
    parser.add_argument(
        "--enhance-only",
        help="Path to existing JSON file to enhance (skip Pass 1)",
    )

    args = parser.parse_args()

    model_names = {
        0: "valhalla/t5-small-qa-qg-hl",
        1: "google/flan-t5-base",
    }

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

        avg_score = sum(quality_stats["quality_scores"]) / max(
            len(quality_stats["quality_scores"]), 1
        )

        enhanced_data = {
            "qa_pairs": enhanced_pairs,
            "total_pairs": len(enhanced_pairs),
            "metadata": {
                **data.get("metadata", {}),
                "enhancement_applied": True,
                "original_count": len(qa_pairs),
                "filtered_count": quality_stats["filtered_out"],
            },
            "quality_stats": {
                **data.get("quality_stats", {}),
                "average_quality_score": avg_score,
                "enhancement_stats": quality_stats,
            },
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(enhanced_data, f, indent=4, ensure_ascii=False)

        logger.info("Enhanced QA pairs saved to %s", args.output)

    else:
        try:
            generator = EnhancedQAPairGenerator(
                model_choice=args.model,
                weights_dir=args.weights,
                use_extractive=args.extractive,
            )
            logger.info("Models loaded successfully for Pass 1")

            logger.info("Extracting text from entire PDF...")
            text = generator.extract_text_from_pdf(args.input)

            if not text:
                logger.error("No text extracted from PDF. Please check the file.")
                return

            logger.info("Extracted %d characters from PDF", len(text))

            qa_pairs = generator.generate_qa_pairs_pass1(text)

            if not qa_pairs:
                logger.error(
                    "No QA pairs generated in Pass 1. Check your input file."
                )
                return

            logger.info("Pass 1 completed: %d QA pairs generated", len(qa_pairs))

            # Free GPU memory before enhancement pass
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            enhancer = QAEnhancer()
            enhanced_pairs, quality_stats = enhancer.enhance_qa_pairs(qa_pairs)

            avg_score = sum(quality_stats["quality_scores"]) / max(
                len(quality_stats["quality_scores"]), 1
            )

            final_data = {
                "qa_pairs": enhanced_pairs,
                "total_pairs": len(enhanced_pairs),
                "metadata": {
                    "model_used": model_names[args.model],
                    "extractive_enabled": args.extractive,
                    "passes_completed": 2,
                    "source_file": os.path.basename(args.input),
                    "original_pass1_count": len(qa_pairs),
                    "final_enhanced_count": len(enhanced_pairs),
                    "filtered_count": quality_stats["filtered_out"],
                },
                "quality_stats": {
                    "average_quality_score": avg_score,
                    "total_generated": len(qa_pairs),
                    "total_enhanced": quality_stats["enhanced"],
                    "total_filtered": quality_stats["filtered_out"],
                    "enhancement_stats": quality_stats,
                },
            }

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)

            logger.info("Final QA pairs saved to %s", args.output)

            # Summary
            logger.info("=" * 60)
            logger.info("PROCESSING COMPLETE")
            logger.info("=" * 60)
            logger.info("Source PDF: %s", os.path.basename(args.input))
            logger.info("Model used: %s", model_names[args.model])
            logger.info("Extractive QA: %s", "Enabled" if args.extractive else "Disabled")
            logger.info("Total characters extracted: %d", len(text))
            logger.info("Pass 1 generated: %d pairs", len(qa_pairs))
            logger.info("Pass 2 enhanced: %d pairs", quality_stats["enhanced"])
            logger.info("Pass 2 filtered: %d pairs", quality_stats["filtered_out"])
            logger.info("Final output: %d pairs", len(enhanced_pairs))
            logger.info("Average quality score: %.3f", avg_score)

            if enhanced_pairs:
                sources: Dict[str, int] = {}
                types: Dict[str, int] = {}
                difficulties: Dict[str, int] = {}

                for pair in enhanced_pairs:
                    source = pair.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1
                    q_type = pair.get("question_type", "general")
                    types[q_type] = types.get(q_type, 0) + 1
                    difficulty = pair.get("difficulty_level", "basic")
                    difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

                logger.info("Answer sources: %s", dict(sources))
                logger.info("Question types: %s", dict(types))
                logger.info("Difficulty levels: %s", dict(difficulties))

            logger.info("=" * 60)

        except Exception as e:
            logger.error("Error during processing: %s", e)
            raise


if __name__ == "__main__":
    main()