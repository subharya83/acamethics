import torch
from transformers import (
    pipeline, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
import pdfplumber
import json
import os
import argparse
import re
from typing import List, Dict, Tuple, Optional
import difflib

class TextProcessor:
    """Handles all text cleaning and preprocessing"""
    
    def __init__(self):
        self.encoding_fixes = {
            'Ã¢â‚¬â„¢': "'", 'Ã¢â‚¬Å"': '"', 'Ã¢â‚¬': '"', 'Ã¢â‚¬"': 'â€"', 'Ã¢â‚¬"': 'â€"',
            'Ã¢â‚¬Â¦': '...', 'Ã¢â‚¬Ëœ': "'", 'Ã‚': ' ', 'ÃƒÂ¡': 'Ã¡', 'ÃƒÂ©': 'Ã©',
            'Ã‚ ': ' ', 'Ã‚Â­': '-'
        }
        
        self.noise_patterns = [
            r'Chapter \d+_.*?\.indd \d+.*?\d{2}:\d{2}:\d{2}',
            r'Ganita Prakash \| Grade \d+',
            r'Patterns in Mathematics\s*\d+',
            r'Math\s+Talk\s*',
            r'Try\s+This\s*',
            r'Figure it Out\s*',
            r'^\d+$',  # Page numbers on their own line
            r'^\s*\.\s*\.\s*\.\s*$'  # Dots indicating continuation
        ]
    
    def fix_encoding(self, text: str) -> str:
        """Fix common encoding issues"""
        for bad, good in self.encoding_fixes.items():
            text = text.replace(bad, good)
        return text
    
    def remove_noise(self, text: str) -> str:
        """Remove page headers, footers, and other noise"""
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def preserve_math_sequences(self, text: str) -> str:
        """Identify and preserve mathematical sequences"""
        # Pattern for number sequences
        seq_pattern = r'(\d+(?:,\s*\d+){3,}(?:,\s*\.{3})?)'
        sequences = re.findall(seq_pattern, text)
        
        # Mark sequences to prevent breaking during chunking
        for seq in sequences:
            text = text.replace(seq, f"[MATH_SEQ:{seq}]")
        
        return text
    
    def restore_math_sequences(self, text: str) -> str:
        """Restore mathematical sequences from markers"""
        return re.sub(r'\[MATH_SEQ:(.*?)\]', r'\1', text)
    
    def clean_extracted_text(self, text: str) -> str:
        """Complete text cleaning pipeline"""
        text = self.fix_encoding(text)
        text = self.remove_noise(text)
        text = self.preserve_math_sequences(text)
        return text

class ContentAnalyzer:
    """Analyzes content type and extracts key information"""
    
    def __init__(self):
        self.content_indicators = {
            "definition": ["is defined as", "refers to", "means", "what is", "called"],
            "example": ["for example", "such as", "including", "like", "instance"],
            "sequence": ["sequence", "pattern", "next", "series"],
            "explanation": ["because", "due to", "reason", "why", "how", "explains"],
            "instruction": ["draw", "copy", "find", "calculate", "solve", "can you"],
            "mathematical": [r'\d+(?:,\s*\d+){2,}', "numbers", "triangular", "square", "cube"]
        }
    
    def identify_content_type(self, text: str) -> str:
        """Identify the primary content type of a text chunk"""
        text_lower = text.lower()
        scores = {}
        
        for content_type, indicators in self.content_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator.startswith('r'):  # regex pattern
                    if re.search(indicator[2:-1], text_lower):
                        score += 2
                elif indicator in text_lower:
                    score += 1
            scores[content_type] = score
        
        return max(scores, key=scores.get) if scores else "general"
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key mathematical and educational concepts"""
        concepts = []
        
        # Mathematical terms
        math_terms = re.findall(r'\b(?:triangular|square|cube|prime|even|odd|fibonacci|sequence|pattern|polygon)\b', 
                               text.lower())
        concepts.extend(math_terms)
        
        # Number sequences
        sequences = re.findall(r'\d+(?:,\s*\d+){2,}', text)
        concepts.extend([f"sequence_{seq.replace(' ', '')}" for seq in sequences])
        
        # Capitalized terms (likely definitions)
        cap_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend([term.lower() for term in cap_terms if len(term.split()) <= 3])
        
        return list(set(concepts))

class QuestionValidator:
    """Validates question quality and structure"""
    
    def __init__(self):
        self.min_length = 15
        self.max_length = 200
        self.question_starters = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can', 'do', 'does', 'is', 'are']
    
    def is_valid_question(self, question: str) -> Tuple[bool, List[str]]:
        """Validate question structure and content"""
        issues = []
        
        if not question or len(question.strip()) == 0:
            return False, ["Empty question"]
        
        question = question.strip()
        
        # Check length
        if len(question) < self.min_length:
            issues.append("Too short")
        elif len(question) > self.max_length:
            issues.append("Too long")
        
        # Check if it ends with question mark
        if not question.endswith('?'):
            issues.append("Missing question mark")
        
        # Check if it starts appropriately
        first_word = question.split()[0].lower() if question.split() else ""
        if first_word not in self.question_starters and not any(starter in question.lower()[:20] for starter in self.question_starters):
            issues.append("Doesn't start like a question")
        
        # Check for statement-like structure
        if question.count('.') >= question.count('?'):
            issues.append("Contains statement fragments")
        
        # Check for encoding issues
        if any(char in question for char in ['Ã¢â‚¬', 'Ã‚']):
            issues.append("Contains encoding errors")
        
        # Check for repetitive content
        words = question.lower().split()
        if len(set(words)) < len(words) * 0.6:  # Less than 60% unique words
            issues.append("Too repetitive")
        
        return len(issues) == 0, issues

class AnswerValidator:
    """Validates answer quality and relevance"""
    
    def __init__(self):
        self.min_length = 5
        self.max_length = 300
        self.noise_patterns = [
            r'\d{2}:\d{2}:\d{2}',  # Timestamps
            r'Chapter \d+',        # Chapter references
            r'\.indd',             # InDesign files
            r'^[A-Z\s]{5,}$',      # All caps headers
            r'^\d+$',              # Just numbers
            r'^[^a-zA-Z]*$'        # No letters
        ]
    
    def is_valid_answer(self, answer: str, question: str = "") -> Tuple[bool, List[str]]:
        """Validate answer quality and relevance"""
        issues = []
        
        if not answer or len(answer.strip()) == 0:
            return False, ["Empty answer"]
        
        answer = answer.strip()
        
        # Check length
        if len(answer) < self.min_length:
            issues.append("Too short")
        elif len(answer) > self.max_length:
            issues.append("Too long")
        
        # Check for noise patterns
        for pattern in self.noise_patterns:
            if re.search(pattern, answer):
                issues.append("Contains metadata/noise")
                break
        
        # Check for question repetition
        if question:
            question_words = set(question.lower().split())
            answer_words = answer.lower().split()
            overlap = len(set(answer_words) & question_words)
            if len(answer_words) > 0 and overlap > len(answer_words) * 0.7:
                issues.append("Mostly repeats question")
        
        # Check for meaningful content
        if len(answer.split()) < 3:
            issues.append("Too few words")
        
        return len(issues) == 0, issues

class EnhancedQAPairGenerator:
    """Enhanced version of QA pair generator with better extraction"""
    
    def __init__(self, model_choice: int = 0, weights_dir: str = "weights", use_extractive: bool = True):
        self.weights_dir = weights_dir
        self.model_choice = model_choice
        self.use_extractive = use_extractive
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.question_validator = QuestionValidator()
        self.answer_validator = AnswerValidator()
        
        # Model configurations
        self.model_configs = {
            0: {"name": "valhalla/t5-small-qa-qg-hl", "type": "t5"},
            1: {"name": "facebook/bart-large-cnn", "type": "bart"},
            2: {"name": "google/flan-t5-base", "type": "flan-t5"}
        }
        
        self.qa_pipeline = self.load_qa_model()
        self.extractive_pipeline = self.load_extractive_model() if use_extractive else None
    
    def load_qa_model(self):
        """Load QA generation model with optimized settings"""
        os.makedirs(self.weights_dir, exist_ok=True)
        config = self.model_configs[self.model_choice]
        
        print(f"Loading {config['type'].upper()} model: {config['name']}")
        
        try:
            if config["type"] in ["t5", "flan-t5"]:
                model = T5ForConditionalGeneration.from_pretrained(config["name"], cache_dir=self.weights_dir)
                tokenizer = T5Tokenizer.from_pretrained(config["name"], cache_dir=self.weights_dir)
            elif config["type"] == "bart":
                model = BartForConditionalGeneration.from_pretrained(config["name"], cache_dir=self.weights_dir)
                tokenizer = BartTokenizer.from_pretrained(config["name"], cache_dir=self.weights_dir)
            
            device = 0 if torch.cuda.is_available() else -1
            return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_extractive_model(self):
        """Load extractive QA model"""
        print("Loading extractive QA model...")
        try:
            model_name = "distilbert-base-uncased-distilled-squad"
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, cache_dir=self.weights_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.weights_dir)
            
            device = 0 if torch.cuda.is_available() else -1
            return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
        except Exception as e:
            print(f"Error loading extractive model: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract and clean text from entire PDF"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"Processing all {total_pages} pages from PDF...")
                
                for i, page in enumerate(pdf.pages):
                    if (i + 1) % 10 == 0:  # Progress indicator
                        print(f"Processed {i + 1}/{total_pages} pages...")
                    
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
        
        # Clean the extracted text
        cleaned_text = self.text_processor.clean_extracted_text(text)
        return cleaned_text
    
    def create_smart_chunks(self, text: str, chunk_size: int = 4) -> List[Dict]:
        """Create context chunks with content analysis"""
        sentences = self.split_into_sentences(text)
        chunks = []
        
        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = " ".join(chunk_sentences)
            
            if len(chunk_text.strip()) > 100:
                # Analyze content type
                content_type = self.content_analyzer.identify_content_type(chunk_text)
                key_concepts = self.content_analyzer.extract_key_concepts(chunk_text)
                
                chunks.append({
                    'text': self.text_processor.restore_math_sequences(chunk_text.strip()),
                    'content_type': content_type,
                    'key_concepts': key_concepts,
                    'sentence_range': (i, min(i + chunk_size, len(sentences)))
                })
            
            i += max(1, chunk_size - 1)  # Overlap for context
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Improved sentence splitting"""
        # More sophisticated sentence splitting
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # More refined filtering
            if (len(sentence) > 20 and 
                not re.match(r'^\d+$', sentence) and 
                not re.match(r'^[A-Z\s]{3,}$', sentence) and
                not re.search(r'\d{2}:\d{2}:\d{2}', sentence)):
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def generate_contextual_questions(self, chunk: Dict) -> List[str]:
        """Generate questions based on content type"""
        context = chunk['text']
        content_type = chunk['content_type']
        
        # Content-specific prompts
        prompts = {
            'definition': f"Generate clear questions asking for definitions from: {context}",
            'sequence': f"Create questions about mathematical patterns and sequences in: {context}",
            'explanation': f"Form questions asking why or how things work in: {context}",
            'example': f"Generate questions about examples and applications in: {context}",
            'instruction': f"Create questions that would help understand the instructions in: {context}"
        }
        
        prompt = prompts.get(content_type, f"Generate educational questions from: {context}")
        
        try:
            config = self.model_configs[self.model_choice]
            
            # Optimized generation parameters
            result = self.qa_pipeline(
                prompt,
                max_length=150,
                min_length=20,
                num_return_sequences=2,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.qa_pipeline.tokenizer.eos_token_id if hasattr(self.qa_pipeline.tokenizer, 'eos_token_id') else 0
            )
            
            questions = []
            if isinstance(result, list):
                for res in result:
                    generated_text = res.get('generated_text', '').strip()
                    questions.extend(self._extract_questions_from_text(generated_text))
            else:
                generated_text = result.get('generated_text', '').strip()
                questions.extend(self._extract_questions_from_text(generated_text))
            
            # Validate and filter questions
            valid_questions = []
            for q in questions:
                is_valid, issues = self.question_validator.is_valid_question(q)
                if is_valid:
                    valid_questions.append(q)
            
            return valid_questions[:3]  # Limit to top 3
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Extract individual questions from generated text"""
        questions = []
        
        # Split by question marks
        potential_questions = re.split(r'\?+', text)
        
        for q in potential_questions:
            q = q.strip()
            if q and len(q) > 10:
                # Clean up prefixes
                q = re.sub(r'^(questions?:?\s*)', '', q, flags=re.IGNORECASE)
                q = re.sub(r'^(\d+\.?\s*)', '', q)
                q = q.strip()
                
                if q and not q.endswith('?'):
                    q += '?'
                    
                if len(q) > 15:
                    questions.append(q)
        
        return questions
    
    def generate_best_answer(self, question: str, chunk: Dict) -> Tuple[str, str]:
        """Generate the best possible answer using multiple methods"""
        context = chunk['text']
        answers = []
        sources = []
        
        # Try extractive QA first (usually most reliable)
        if self.extractive_pipeline:
            try:
                result = self.extractive_pipeline(question=question, context=context)
                answer = result.get('answer', '').strip()
                confidence = result.get('score', 0)
                
                if confidence > 0.3 and len(answer) > 5:
                    is_valid, _ = self.answer_validator.is_valid_answer(answer, question)
                    if is_valid:
                        answers.append(answer)
                        sources.append('extractive')
            except Exception:
                pass
        
        # Try generative model
        try:
            config = self.model_configs[self.model_choice]
            
            if config["type"] == "t5":
                input_text = f"answer: {question} context: {context}"
            elif config["type"] == "flan-t5":
                input_text = f"Answer this question based on the context: {question}\nContext: {context}"
            elif config["type"] == "bart":
                input_text = f"Question: {question}\nContext: {context}\nAnswer:"
            
            result = self.qa_pipeline(
                input_text,
                max_length=100,
                min_length=10,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.qa_pipeline.tokenizer.eos_token_id if hasattr(self.qa_pipeline.tokenizer, 'eos_token_id') else 0
            )
            
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '').strip()
            else:
                answer = result.get('generated_text', '').strip()
            
            # Clean answer
            answer = self._clean_answer(answer, question)
            
            if answer:
                is_valid, _ = self.answer_validator.is_valid_answer(answer, question)
                if is_valid:
                    answers.append(answer)
                    sources.append('generative')
        
        except Exception:
            pass
        
        # Choose best answer (prefer extractive, then generative)
        if answers:
            if 'extractive' in sources:
                idx = sources.index('extractive')
                return answers[idx], sources[idx]
            else:
                return answers[0], sources[0]
        
        return "", "none"
    
    def _clean_answer(self, answer: str, question: str) -> str:
        """Clean and validate generated answers"""
        # Remove common prefixes
        answer = re.sub(r'^(answer:?\s*)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'^(the answer is:?\s*)', '', answer, flags=re.IGNORECASE)
        
        # Remove question repetition check
        question_words = set(question.lower().split())
        answer_words = answer.lower().split()
        
        if len(answer_words) > 0:
            overlap = len(set(answer_words) & question_words)
            if overlap > len(answer_words) * 0.7:
                return ""
        
        return answer.strip()
    
    def generate_qa_pairs_pass1(self, text: str) -> List[Dict[str, str]]:
        """First pass: Generate QA pairs with enhanced extraction"""
        print("Pass 1: Enhanced QA pair generation...")
        
        chunks = self.create_smart_chunks(text, chunk_size=5)
        if len(chunks) < 2:
            print("Not enough content chunks for meaningful QA pairs")
            return []
        
        print(f"Created {len(chunks)} content-aware chunks")
        qa_pairs = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} (Type: {chunk['content_type']})")
            
            questions = self.generate_contextual_questions(chunk)
            
            for question in questions:
                answer, source = self.generate_best_answer(question, chunk)
                
                if answer:
                    qa_pair = {
                        "question": question,
                        "answer": answer,
                        "context": chunk['text'],
                        "source": source,
                        "content_type": chunk['content_type'],
                        "key_concepts": chunk['key_concepts'],
                        "model_used": self.model_configs[self.model_choice]["name"],
                        "quality_score": self._calculate_quality_score(question, answer, chunk['text'])
                    }
                    qa_pairs.append(qa_pair)
        
        print(f"Generated {len(qa_pairs)} QA pairs in Pass 1")
        return qa_pairs
    
    def _calculate_quality_score(self, question: str, answer: str, context: str) -> float:
        """Calculate quality score for QA pair"""
        score = 0.0
        
        # Question quality
        is_valid_q, _ = self.question_validator.is_valid_question(question)
        if is_valid_q:
            score += 0.3
        
        # Answer quality
        is_valid_a, _ = self.answer_validator.is_valid_answer(answer, question)
        if is_valid_a:
            score += 0.3
        
        # Semantic relevance (simple keyword overlap)
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        c_words = set(context.lower().split())
        
        qa_overlap = len(q_words & a_words) / max(len(q_words), 1)
        ac_overlap = len(a_words & c_words) / max(len(a_words), 1)
        
        score += min(0.2, qa_overlap * 0.4)  # Some overlap good, too much bad
        score += min(0.2, ac_overlap * 0.4)  # Answer should relate to context
        
        return min(1.0, score)

class QAEnhancer:
    """Second pass: Enhance existing QA pairs JSON"""
    
    def __init__(self):
        self.question_validator = QuestionValidator()
        self.answer_validator = AnswerValidator()
        self.text_processor = TextProcessor()
    
    def enhance_qa_pairs(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Enhance QA pairs list and return stats"""
        print("Pass 2: Enhancing QA pairs...")
        
        enhanced_pairs = []
        quality_stats = {'filtered_out': 0, 'enhanced': 0, 'quality_scores': []}
        
        for i, pair in enumerate(qa_pairs):
            enhanced_pair = self._enhance_single_pair(pair)
            
            if enhanced_pair:
                enhanced_pairs.append(enhanced_pair)
                quality_stats['enhanced'] += 1
                quality_stats['quality_scores'].append(enhanced_pair.get('quality_score', 0))
            else:
                quality_stats['filtered_out'] += 1
        
        print(f"Enhanced: {quality_stats['enhanced']}, Filtered out: {quality_stats['filtered_out']}")
        return enhanced_pairs, quality_stats
    
    def _enhance_single_pair(self, pair: Dict) -> Optional[Dict]:
        """Enhance a single QA pair"""
        question = pair.get('question', '').strip()
        answer = pair.get('answer', '').strip()
        context = pair.get('context', '').strip()
        
        # Clean text elements
        question = self.text_processor.fix_encoding(question)
        answer = self.text_processor.fix_encoding(answer)
        context = self.text_processor.fix_encoding(context)
        
        # Validate question
        is_valid_q, q_issues = self.question_validator.is_valid_question(question)
        if not is_valid_q:
            return None
        
        # Validate answer
        is_valid_a, a_issues = self.answer_validator.is_valid_answer(answer, question)
        if not is_valid_a:
            return None
        
        # Calculate quality score
        quality_score = self._calculate_enhanced_quality_score(question, answer, context)
        
        # Filter out low quality pairs
        if quality_score < 0.4:
            return None
        
        # Enhance the pair
        enhanced_pair = {
            **pair,
            'question': question,
            'answer': answer,
            'context': context,
            'quality_score': quality_score,
            'question_length': len(question),
            'answer_length': len(answer),
            'question_type': self._classify_question_type(question),
            'difficulty_level': self._assess_difficulty(question, answer),
            'topic_keywords': self._extract_topic_keywords(context)
        }
        
        return enhanced_pair
    
    def _calculate_enhanced_quality_score(self, question: str, answer: str, context: str) -> float:
        """Enhanced quality scoring"""
        score = 0.0
        
        # Length appropriateness
        if 20 <= len(question) <= 150:
            score += 0.2
        if 10 <= len(answer) <= 200:
            score += 0.2
        
        # Structural quality
        if question.count('?') == 1 and question.endswith('?'):
            score += 0.1
        
        # Content quality
        q_words = question.lower().split()
        a_words = answer.lower().split()
        c_words = context.lower().split()
        
        # Reasonable overlap between answer and context
        ac_overlap = len(set(a_words) & set(c_words)) / max(len(a_words), 1)
        if 0.2 <= ac_overlap <= 0.8:
            score += 0.2
        
        # Question-answer relevance (some overlap, not too much)
        qa_overlap = len(set(q_words) & set(a_words)) / max(len(q_words), 1)
        if 0.1 <= qa_overlap <= 0.5:
            score += 0.15
        
        # Diversity in vocabulary
        unique_ratio = len(set(a_words)) / max(len(a_words), 1)
        if unique_ratio > 0.7:
            score += 0.15
        
        return min(1.0, score)
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        q_lower = question.lower()
        
        if q_lower.startswith(('what is', 'what are', 'what does')):
            return 'definition'
        elif q_lower.startswith(('how', 'how can', 'how do')):
            return 'explanation'
        elif q_lower.startswith(('why', 'why do', 'why does')):
            return 'reasoning'
        elif 'sequence' in q_lower or 'pattern' in q_lower:
            return 'mathematical_pattern'
        elif 'example' in q_lower or 'such as' in q_lower:
            return 'example'
        else:
            return 'general'
    
    def _assess_difficulty(self, question: str, answer: str) -> str:
        """Assess difficulty level"""
        # Simple heuristic based on length and complexity
        q_words = len(question.split())
        a_words = len(answer.split())
        
        # Check for complex mathematical terms
        complex_terms = ['theorem', 'proof', 'algorithm', 'derivative', 'integral', 'matrix']
        has_complex = any(term in question.lower() + answer.lower() for term in complex_terms)
        
        # Check for multiple concepts
        concept_indicators = ['and', 'relationship', 'compare', 'analyze', 'explain why']
        multi_concept = any(ind in question.lower() for ind in concept_indicators)
        
        if has_complex or multi_concept or (q_words > 20 and a_words > 30):
            return 'advanced'
        elif q_words > 15 or a_words > 20:
            return 'intermediate'
        else:
            return 'basic'
    
    def _extract_topic_keywords(self, context: str) -> List[str]:
        """Extract topic-relevant keywords"""
        # Mathematical and educational keywords
        topic_patterns = [
            r'\b(?:pattern|sequence|number|triangle|square|polygon|geometry|mathematics|equation)\b',
            r'\b(?:definition|example|explanation|theory|concept|principle)\b',
            r'\b(?:calculate|solve|find|determine|identify|analyze)\b'
        ]
        
        keywords = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, context.lower())
            keywords.extend(matches)
        
        return list(set(keywords))

def main():
    """Main function to run both passes and generate single output file"""
    parser = argparse.ArgumentParser(description="Two-pass QA pair generator with enhancement")
    parser.add_argument("-i", "--input", required=True, help="Path to input PDF file")
    parser.add_argument("-o", "--output", required=True, help="Path for output JSON file")
    parser.add_argument("-w", "--weights", default="weights", help="Directory to store model weights")
    parser.add_argument("-m", "--model", type=int, choices=[0, 1, 2], default=0,
                        help="Model choice: 0=T5-QA-QG, 1=BART-CNN, 2=FLAN-T5")
    parser.add_argument("-x", "--extractive", action="store_true", 
                        help="Enable extractive QA model")
    parser.add_argument("--enhance-only", help="Path to existing JSON file to enhance (skip Pass 1)")
    
    args = parser.parse_args()
    
    model_names = {
        0: "valhalla/t5-small-qa-qg-hl",
        1: "facebook/bart-large-cnn", 
        2: "google/flan-t5-base"
    }
    
    print(f"Selected model: {model_names[args.model]}")
    print(f"Extractive QA: {'ENABLED' if args.extractive else 'DISABLED'}")
    
    if args.enhance_only:
        # Only run Pass 2 (enhancement)
        print(f"Enhancement-only mode: processing {args.enhance_only}")
        
        # Load existing JSON
        with open(args.enhance_only, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = data.get('qa_pairs', [])
        print(f"Loaded {len(qa_pairs)} existing QA pairs")
        
        # Enhance the pairs
        enhancer = QAEnhancer()
        enhanced_pairs, quality_stats = enhancer.enhance_qa_pairs(qa_pairs)
        
        # Create final output
        enhanced_data = {
            "qa_pairs": enhanced_pairs,
            "total_pairs": len(enhanced_pairs),
            "metadata": {
                **data.get('metadata', {}),
                "enhancement_applied": True,
                "original_count": len(qa_pairs),
                "filtered_count": quality_stats['filtered_out']
            },
            "quality_stats": {
                **data.get('quality_stats', {}),
                "average_quality_score": sum(quality_stats['quality_scores']) / max(len(quality_stats['quality_scores']), 1),
                "enhancement_stats": quality_stats
            }
        }
        
        # Save enhanced JSON
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=4, ensure_ascii=False)
        
        print(f"Enhanced QA pairs saved to {args.output}")
        
    else:
        # Run both passes
        try:
            # Pass 1: Enhanced generation
            generator = EnhancedQAPairGenerator(
                model_choice=args.model, 
                weights_dir=args.weights, 
                use_extractive=args.extractive
            )
            print("Models loaded successfully for Pass 1")
            
            # Extract and process text from entire PDF
            print("Extracting text from entire PDF...")
            text = generator.extract_text_from_pdf(args.input)
            
            if not text:
                print("No text extracted from PDF. Please check the file.")
                return
            
            print(f"Extracted {len(text)} characters from PDF")
            
            # Generate QA pairs (Pass 1)
            qa_pairs = generator.generate_qa_pairs_pass1(text)
            
            if not qa_pairs:
                print("No QA pairs generated in Pass 1. Please check your input file.")
                return
            
            print(f"Pass 1 completed: {len(qa_pairs)} QA pairs generated")
            
            # Pass 2: Enhancement
            enhancer = QAEnhancer()
            enhanced_pairs, quality_stats = enhancer.enhance_qa_pairs(qa_pairs)
            
            # Create final output data structure
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
                    "filtered_count": quality_stats['filtered_out']
                },
                "quality_stats": {
                    "average_quality_score": sum(quality_stats['quality_scores']) / max(len(quality_stats['quality_scores']), 1),
                    "total_generated": len(qa_pairs),
                    "total_enhanced": quality_stats['enhanced'],
                    "total_filtered": quality_stats['filtered_out'],
                    "enhancement_stats": quality_stats
                }
            }
            
            # Save final JSON
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
            
            print(f"Final QA pairs saved to {args.output}")
            
            # Print comprehensive summary
            print("\n" + "="*60)
            print("PROCESSING COMPLETE - FINAL SUMMARY")
            print("="*60)
            print(f"Source PDF: {os.path.basename(args.input)}")
            print(f"Model used: {model_names[args.model]}")
            print(f"Extractive QA: {'Enabled' if args.extractive else 'Disabled'}")
            print(f"Total characters extracted: {len(text):,}")
            print()
            print("PASS 1 RESULTS:")
            print(f"  Generated QA pairs: {len(qa_pairs)}")
            print()
            print("PASS 2 RESULTS:")
            print(f"  Enhanced pairs: {quality_stats['enhanced']}")
            print(f"  Filtered out: {quality_stats['filtered_out']}")
            print(f"  Final pairs in output: {len(enhanced_pairs)}")
            print(f"  Average quality score: {final_data['quality_stats']['average_quality_score']:.3f}")
            
            # Quality breakdown
            if enhanced_pairs:
                sources = {}
                types = {}
                difficulties = {}
                
                for pair in enhanced_pairs:
                    # Source breakdown
                    source = pair.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                    
                    # Question type breakdown
                    q_type = pair.get('question_type', 'general')
                    types[q_type] = types.get(q_type, 0) + 1
                    
                    # Difficulty breakdown
                    difficulty = pair.get('difficulty_level', 'basic')
                    difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                
                print()
                print("QUALITY BREAKDOWN:")
                print(f"  Answer sources: {dict(sources)}")
                print(f"  Question types: {dict(types)}")
                print(f"  Difficulty levels: {dict(difficulties)}")
            
            print(f"\nFinal output file: {args.output}")
            print("="*60)
            
        except Exception as e:
            print(f"Error during processing: {e}")
            raise

if __name__ == "__main__":
    main()