import torch
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import pdfplumber
import json
import os
import argparse
import re
from typing import List, Dict, Tuple

class QAPairGenerator:
    def __init__(self, weights_dir="weights"):
        self.weights_dir = weights_dir
        self.qa_pipeline = self.load_qa_model()
        
    def load_qa_model(self):
        """Load the QA model locally"""
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Using a model that's better for QA pair generation
        model_name = "valhalla/t5-small-qa-qg-hl"
        model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=self.weights_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=self.weights_dir)
        
        device = 0 if torch.cuda.is_available() else -1
        qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
        return qa_pipeline
    
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
        # Simple sentence splitting - you might want to use spacy or nltk for better results
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def create_context_chunks(self, sentences: List[str], chunk_size: int = 3) -> List[str]:
        """Create context chunks from sentences"""
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only include substantial chunks
                chunks.append(chunk.strip())
        return chunks
    
    def generate_questions_from_context(self, context: str) -> List[str]:
        """Generate questions from a given context"""
        try:
            # Format input for question generation
            input_text = f"generate questions: {context}"
            
            # Generate questions
            result = self.qa_pipeline(
                input_text,
                max_length=256,
                num_return_sequences=3,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.qa_pipeline.tokenizer.eos_token_id
            )
            
            questions = []
            for res in result:
                generated_text = res['generated_text'].strip()
                # Clean up the generated text and extract questions
                if generated_text:
                    # Split multiple questions if they exist
                    question_list = re.split(r'[?]\s*', generated_text)
                    for q in question_list:
                        q = q.strip()
                        if q and not q.endswith('?'):
                            q += '?'
                        if len(q) > 10 and q.endswith('?'):
                            questions.append(q)
            
            return questions[:2]  # Limit to 2 questions per context
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def generate_answer_from_context(self, question: str, context: str) -> str:
        """Generate answer for a question given context"""
        try:
            # Format input for answer generation
            input_text = f"question: {question} context: {context}"
            
            result = self.qa_pipeline(
                input_text,
                max_length=128,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.qa_pipeline.tokenizer.eos_token_id
            )
            
            if result and len(result) > 0:
                answer = result[0]['generated_text'].strip()
                return answer if answer else "No answer generated"
            else:
                return "No answer generated"
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error generating answer"
    
    def extract_answer_from_context(self, question: str, context: str) -> str:
        """Extract answer from context using simple keyword matching"""
        try:
            # Convert to lowercase for matching
            q_lower = question.lower()
            context_lower = context.lower()
            
            # Simple extraction based on question type
            if q_lower.startswith(('who', 'what', 'where', 'when', 'why', 'how')):
                # Find the sentence that might contain the answer
                sentences = re.split(r'[.!?]+', context)
                for sentence in sentences:
                    # Look for sentences that might contain relevant keywords
                    sentence_lower = sentence.lower()
                    question_words = [word for word in q_lower.split() if len(word) > 3]
                    
                    matches = sum(1 for word in question_words if word in sentence_lower)
                    if matches >= 2:  # At least 2 key words match
                        return sentence.strip()
            
            # Fallback: return first substantial sentence
            sentences = re.split(r'[.!?]+', context)
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    return sentence.strip()
                    
            return "Answer not found in context"
            
        except Exception as e:
            return f"Error extracting answer: {e}"
    
    def generate_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """Generate question-answer pairs from text"""
        print("Processing text and generating QA pairs...")
        
        # Split text into sentences and create chunks
        sentences = self.split_into_sentences(text)
        if len(sentences) < 3:
            print("Not enough content to generate meaningful QA pairs")
            return []
        
        chunks = self.create_context_chunks(sentences, chunk_size=4)
        qa_pairs = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Generate questions for this chunk
            questions = self.generate_questions_from_context(chunk)
            
            for question in questions:
                # Try to generate answer using the model
                model_answer = self.generate_answer_from_context(question, chunk)
                
                # If model answer is not good, try extraction
                if (model_answer in ["No answer generated", "Error generating answer"] or 
                    len(model_answer) < 10):
                    extracted_answer = self.extract_answer_from_context(question, chunk)
                    final_answer = extracted_answer
                else:
                    final_answer = model_answer
                
                # Create QA pair
                qa_pair = {
                    "question": question,
                    "answer": final_answer,
                    "context": chunk,
                    "source": "generated"
                }
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def save_qa_pairs(self, qa_pairs: List[Dict[str, str]], output_file: str):
        """Save QA pairs to a JSON file"""
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump({
                    "qa_pairs": qa_pairs,
                    "total_pairs": len(qa_pairs),
                    "metadata": {
                        "model_used": "valhalla/t5-small-qa-qg-hl",
                        "generation_method": "context-based"
                    }
                }, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(qa_pairs)} QA pairs to {output_file}")
        except Exception as e:
            print(f"Error saving QA pairs: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from a PDF file for SLM fine-tuning.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file.")
    parser.add_argument("-w", "--weights", default="weights", help="Directory to store model weights.")
    args = parser.parse_args()
    
    # Initialize the QA pair generator
    generator = QAPairGenerator(weights_dir=args.weights)
    print("QA model loaded successfully.")
    
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

if __name__ == "__main__":
    main()