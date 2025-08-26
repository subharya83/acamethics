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

class QAPairGenerator:
    def __init__(self, model_choice: int = 0, weights_dir: str = "weights", use_extractive: bool = False):
        self.weights_dir = weights_dir
        self.model_choice = model_choice
        self.use_extractive = use_extractive
        
        # Model configurations
        self.model_configs = {
            0: {
                "name": "valhalla/t5-small-qa-qg-hl",
                "type": "t5",
                "description": "T5 Small QA-QG Model"
            },
            1: {
                "name": "facebook/bart-large-cnn",
                "type": "bart",
                "description": "BART Large CNN Model"
            },
            2: {
                "name": "google/flan-t5-base",
                "type": "flan-t5",
                "description": "FLAN-T5 Base Model"
            }
        }
        
        # Load models
        self.qa_pipeline = self.load_qa_model()
        self.extractive_pipeline = self.load_extractive_model() if use_extractive else None
        
    def load_qa_model(self):
        """Load the selected QA generation model"""
        os.makedirs(self.weights_dir, exist_ok=True)
        
        config = self.model_configs[self.model_choice]
        model_name = config["name"]
        model_type = config["type"]
        
        print(f"Loading {config['description']} ({model_name})...")
        
        try:
            if model_type == "t5" or model_type == "flan-t5":
                model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=self.weights_dir)
                tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=self.weights_dir)
                task = "text2text-generation"
                
            elif model_type == "bart":
                model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=self.weights_dir)
                tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=self.weights_dir)
                task = "text2text-generation"
            
            device = 0 if torch.cuda.is_available() else -1
            pipeline_obj = pipeline(task, model=model, tokenizer=tokenizer, device=device)
            
            print(f"Successfully loaded {config['description']}")
            return pipeline_obj
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    
    def load_extractive_model(self):
        """Load extractive QA model for better answer extraction"""
        print("Loading extractive QA model...")
        
        try:
            # Using DistilBERT for extractive QA
            extractive_model_name = "distilbert-base-uncased-distilled-squad"
            
            # Explicitly load model and tokenizer to ensure they're cached in weights_dir
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            
            print(f"Downloading/loading {extractive_model_name} model...")
            extractive_model = AutoModelForQuestionAnswering.from_pretrained(
                extractive_model_name, 
                cache_dir=self.weights_dir
            )
            
            print(f"Downloading/loading {extractive_model_name} tokenizer...")
            extractive_tokenizer = AutoTokenizer.from_pretrained(
                extractive_model_name, 
                cache_dir=self.weights_dir
            )
            
            # Create pipeline with the explicitly loaded model and tokenizer
            extractive_pipeline = pipeline(
                "question-answering",
                model=extractive_model,
                tokenizer=extractive_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("Successfully loaded extractive QA model")
            return extractive_pipeline
            
        except Exception as e:
            print(f"Error loading extractive model: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better processing"""
        # Improved sentence splitting
        sentences = re.split(r'[.!?]+(?:\s+|$)', text)
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short sentences, page numbers, headers, etc.
            if (len(sentence) > 15 and 
                not re.match(r'^\d+$', sentence) and  # Not just numbers
                not re.match(r'^[A-Z\s]{3,}$', sentence)):  # Not all caps headers
                cleaned_sentences.append(sentence)
                
        return cleaned_sentences
    
    def create_context_chunks(self, sentences: List[str], chunk_size: int = 4) -> List[str]:
        """Create context chunks from sentences with overlap"""
        chunks = []
        overlap = 1  # Overlap between chunks for context continuity
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk = " ".join(chunk_sentences)
            
            # Only include substantial chunks
            if len(chunk.strip()) > 100:
                chunks.append(chunk.strip())
                
        return chunks
    
    def generate_questions_from_context(self, context: str) -> List[str]:
        """Generate questions from context using the selected model"""
        try:
            config = self.model_configs[self.model_choice]
            questions = []
            
            if config["type"] == "t5":
                # T5 format for question generation
                input_text = f"generate questions: {context}"
                
            elif config["type"] == "flan-t5":
                # FLAN-T5 format with instruction
                input_text = f"Generate 2-3 questions based on this text: {context}"
                
            elif config["type"] == "bart":
                # BART format - we'll use it for question generation too
                input_text = f"Generate questions about: {context}"
            
            # Generate questions
            result = self.qa_pipeline(
                input_text,
                max_length=200,
                min_length=10,
                num_return_sequences=3 if config["type"] != "bart" else 1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.qa_pipeline.tokenizer.eos_token_id if hasattr(self.qa_pipeline.tokenizer, 'eos_token_id') else 0
            )
            
            # Process results
            if isinstance(result, list):
                for res in result:
                    generated_text = res.get('generated_text', '').strip()
                    questions.extend(self._extract_questions_from_text(generated_text))
            else:
                generated_text = result.get('generated_text', '').strip()
                questions.extend(self._extract_questions_from_text(generated_text))
            
            # Filter and clean questions
            clean_questions = self._filter_questions(questions)
            return clean_questions[:3]  # Limit to 3 questions per context
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Extract individual questions from generated text"""
        questions = []
        
        # Split by question marks and clean up
        potential_questions = re.split(r'\?+', text)
        
        for q in potential_questions:
            q = q.strip()
            if q and len(q) > 10:
                # Clean up common prefixes
                q = re.sub(r'^(questions?:?\s*)', '', q, flags=re.IGNORECASE)
                q = re.sub(r'^(\d+\.?\s*)', '', q)  # Remove numbering
                q = q.strip()
                
                if q and not q.endswith('?'):
                    q += '?'
                    
                if len(q) > 15:  # Ensure substantial questions
                    questions.append(q)
                    
        return questions
    
    def _filter_questions(self, questions: List[str]) -> List[str]:
        """Filter out low-quality questions"""
        filtered = []
        
        for q in questions:
            q = q.strip()
            
            # Skip very short questions
            if len(q) < 15:
                continue
                
            # Skip questions that are too generic
            generic_patterns = [
                r'^what is (this|that|it)\?$',
                r'^how (is|are) (this|that|it)\?$',
                r'^why (is|are) (this|that|it)\?$'
            ]
            
            is_generic = any(re.match(pattern, q.lower()) for pattern in generic_patterns)
            if is_generic:
                continue
                
            # Skip duplicate questions
            if q not in filtered:
                filtered.append(q)
                
        return filtered
    
    def generate_answer_with_model(self, question: str, context: str) -> str:
        """Generate answer using the generative model"""
        try:
            config = self.model_configs[self.model_choice]
            
            if config["type"] == "t5":
                input_text = f"question: {question} context: {context}"
            elif config["type"] == "flan-t5":
                input_text = f"Answer this question based on the context: {question}\nContext: {context}"
            elif config["type"] == "bart":
                input_text = f"Question: {question}\nContext: {context}\nAnswer:"
            
            result = self.qa_pipeline(
                input_text,
                max_length=150,
                min_length=5,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.qa_pipeline.tokenizer.eos_token_id if hasattr(self.qa_pipeline.tokenizer, 'eos_token_id') else 0
            )
            
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '').strip()
            else:
                answer = result.get('generated_text', '').strip()
                
            # Clean up the answer
            answer = self._clean_answer(answer, question)
            return answer if answer else "No answer generated"
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error generating answer"
    
    def generate_extractive_answer(self, question: str, context: str) -> str:
        """Generate answer using extractive QA model"""
        if not self.extractive_pipeline:
            return "Extractive model not available"
            
        try:
            result = self.extractive_pipeline(question=question, context=context)
            
            answer = result.get('answer', '').strip()
            confidence = result.get('score', 0)
            
            # Only return high-confidence answers
            if confidence > 0.1 and len(answer) > 3:
                return answer
            else:
                return "Low confidence answer"
                
        except Exception as e:
            print(f"Error with extractive QA: {e}")
            return "Extractive QA failed"
    
    def _clean_answer(self, answer: str, question: str) -> str:
        """Clean and validate generated answers"""
        # Remove common prefixes
        answer = re.sub(r'^(answer:?\s*)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'^(the answer is:?\s*)', '', answer, flags=re.IGNORECASE)
        
        # Remove question repetition
        question_words = set(question.lower().split())
        answer_words = answer.lower().split()
        
        # If answer just repeats the question, it's not useful
        overlap = len(set(answer_words) & question_words)
        if overlap > len(answer_words) * 0.7:
            return ""
            
        return answer.strip()
    
    def extract_answer_from_context(self, question: str, context: str) -> str:
        """Extract answer from context using keyword matching and heuristics"""
        try:
            # Convert to lowercase for matching
            q_lower = question.lower()
            context_sentences = re.split(r'[.!?]+', context)
            
            # Extract key terms from question (excluding common words)
            stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'the', 'a', 'an'}
            question_terms = [word.lower() for word in re.findall(r'\b\w+\b', question) 
                            if len(word) > 2 and word.lower() not in stop_words]
            
            best_sentence = ""
            max_score = 0
            
            for sentence in context_sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                    
                sentence_lower = sentence.lower()
                
                # Score based on term overlap
                score = sum(1 for term in question_terms if term in sentence_lower)
                
                # Boost score for sentences with specific patterns
                if any(pattern in sentence_lower for pattern in ['because', 'due to', 'result in', 'caused by']):
                    score += 1
                    
                if score > max_score and len(sentence) > 20:
                    max_score = score
                    best_sentence = sentence
            
            return best_sentence if best_sentence else "No suitable answer found in context"
            
        except Exception as e:
            return f"Error extracting answer: {e}"
    
    def generate_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """Generate question-answer pairs from text"""
        print("Processing text and generating QA pairs...")
        
        # Split text into sentences and create chunks
        sentences = self.split_into_sentences(text)
        if len(sentences) < 5:
            print("Not enough content to generate meaningful QA pairs")
            return []
        
        print(f"Found {len(sentences)} sentences, creating context chunks...")
        chunks = self.create_context_chunks(sentences, chunk_size=5)
        qa_pairs = []
        
        config = self.model_configs[self.model_choice]
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Generate questions for this chunk
            questions = self.generate_questions_from_context(chunk)
            
            for question in questions:
                answers = []
                answer_sources = []
                
                # Try generative model
                model_answer = self.generate_answer_with_model(question, chunk)
                if model_answer not in ["No answer generated", "Error generating answer"] and len(model_answer) > 10:
                    answers.append(model_answer)
                    answer_sources.append("generative")
                
                # Try extractive model if enabled
                if self.use_extractive:
                    extractive_answer = self.generate_extractive_answer(question, chunk)
                    if extractive_answer not in ["Extractive model not available", "Low confidence answer", "Extractive QA failed"]:
                        answers.append(extractive_answer)
                        answer_sources.append("extractive")
                
                # Fallback to keyword extraction
                if not answers:
                    extracted_answer = self.extract_answer_from_context(question, chunk)
                    answers.append(extracted_answer)
                    answer_sources.append("keyword_extraction")
                
                # Choose the best answer (prefer extractive if available, then generative)
                if len(answers) > 1 and "extractive" in answer_sources:
                    final_answer = answers[answer_sources.index("extractive")]
                    final_source = "extractive"
                elif "generative" in answer_sources:
                    final_answer = answers[answer_sources.index("generative")]
                    final_source = "generative"
                else:
                    final_answer = answers[0]
                    final_source = answer_sources[0]
                
                # Create QA pair
                qa_pair = {
                    "question": question,
                    "answer": final_answer,
                    "context": chunk,
                    "source": final_source,
                    "model_used": config["description"],
                    "all_answers": dict(zip(answer_sources, answers)) if len(answers) > 1 else None
                }
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def save_qa_pairs(self, qa_pairs: List[Dict[str, str]], output_file: str):
        """Save QA pairs to a JSON file"""
        try:
            config = self.model_configs[self.model_choice]
            
            output_data = {
                "qa_pairs": qa_pairs,
                "total_pairs": len(qa_pairs),
                "metadata": {
                    "primary_model": config["description"],
                    "model_name": config["name"],
                    "extractive_qa_enabled": self.use_extractive,
                    "extractive_model": "distilbert-base-uncased-distilled-squad" if self.use_extractive else None,
                    "generation_method": "multi-model" if self.use_extractive else "single-model"
                },
                "quality_stats": self._calculate_quality_stats(qa_pairs)
            }
            
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
                
            print(f"Successfully saved {len(qa_pairs)} QA pairs to {output_file}")
            
        except Exception as e:
            print(f"Error saving QA pairs: {e}")
    
    def _calculate_quality_stats(self, qa_pairs: List[Dict[str, str]]) -> Dict:
        """Calculate quality statistics for the generated QA pairs"""
        if not qa_pairs:
            return {}
            
        sources = [pair.get("source", "unknown") for pair in qa_pairs]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        avg_question_length = sum(len(pair.get("question", "")) for pair in qa_pairs) / len(qa_pairs)
        avg_answer_length = sum(len(pair.get("answer", "")) for pair in qa_pairs) / len(qa_pairs)
        
        return {
            "answer_sources": source_counts,
            "average_question_length": round(avg_question_length, 2),
            "average_answer_length": round(avg_answer_length, 2),
            "pairs_with_multiple_answers": len([p for p in qa_pairs if p.get("all_answers")])
        }

def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from a PDF file for SLM fine-tuning.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file.")
    parser.add_argument("-w", "--weights", default="weights", help="Directory to store model weights.")
    parser.add_argument("-m", "--model", type=int, choices=[0, 1, 2], default=0,
                        help="Model choice: 0=valhalla/t5-small-qa-qg-hl, 1=facebook/bart-large-cnn, 2=google/flan-t5-base")
    parser.add_argument("-x", "--extractive", action="store_true", 
                        help="Enable extractive QA model for better answer generation")
    
    args = parser.parse_args()
    
    # Print model selection
    model_names = {
        0: "valhalla/t5-small-qa-qg-hl",
        1: "facebook/bart-large-cnn", 
        2: "google/flan-t5-base"
    }
    
    print(f"Selected model: {model_names[args.model]}")
    if args.extractive:
        print("Extractive QA model: ENABLED")
    else:
        print("Extractive QA model: DISABLED")
    
    # Initialize the QA pair generator
    try:
        generator = QAPairGenerator(
            model_choice=args.model, 
            weights_dir=args.weights, 
            use_extractive=args.extractive
        )
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    text = generator.extract_text_from_pdf(args.input)
    
    if not text:
        print("No text extracted from PDF. Please check the file.")
        return
    
    print(f"Extracted {len(text)} characters from PDF.")
    
    # Generate QA pairs
    qa_pairs = generator.generate_qa_pairs(text)
    
    if not qa_pairs:
        print("No QA pairs generated. Please check your input file.")
        return
    
    # Save QA pairs
    generator.save_qa_pairs(qa_pairs, args.output)
    print(f"Process completed! Generated {len(qa_pairs)} QA pairs.")
    
    # Print summary
    sources = {}
    for pair in qa_pairs:
        source = pair.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    print("\nGeneration Summary:")
    for source, count in sources.items():
        print(f"  {source}: {count} pairs")

if __name__ == "__main__":
    main()