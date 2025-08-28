# Acamethics

This project provides a comprehensive set of tools for generating high-quality 
question-answer pairs from PDF documents and fine-tuning language models for 
specialized question-answering tasks. The system consists of three main 
components with enhanced processing capabilities.

## Components

1. **Enhanced PDF QA Generator (genQA.py)**
   - Two-pass QA generation system with content analysis
   - Extracts text from entire PDF documents with intelligent preprocessing
   - Content-aware chunking with mathematical sequence preservation
   - Multi-model support with both generative and extractive QA
   - Advanced quality validation and enhancement
   - Comprehensive statistics and quality scoring

2. **Model Fine-tuning (fineTuneSLM.py)**
   - Fine-tunes DistilBERT models for question-answering tasks
   - Supports resume training from checkpoints
   - Optimized for various device types (CUDA, MPS, CPU)
   - Advanced training arguments with evaluation metrics
   - Automatic model saving and inference testing

3. **Query Interface (querySLM.py)**
   - Supports both CLI and web server modes
   - T5-based answer generation with beam search
   - Batch processing capabilities
   - Simple web GUI for interactive querying

## Installation

Install the required packages:
```bash
pip install torch transformers pdfplumber datasets sentencepiece scikit-learn tqdm flask
```

## Usage

### 1. Enhanced QA Pair Generation from PDF

The enhanced QA generator now supports two operational modes and significantly improved quality.

```bash
usage: genQA.py [-h] -i INPUT -o OUTPUT [-w WEIGHTS] [-m {0,1,2}] [-x] [--enhance-only ENHANCE_ONLY]

Two-pass QA pair generator with enhancement

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input PDF file
  -o OUTPUT, --output OUTPUT
                        Path for output JSON file
  -w WEIGHTS, --weights WEIGHTS
                        Directory to store model weights (default: weights)
  -m {0,1,2}, --model {0,1,2}
                        Model choice: 0=T5-QA-QG, 1=BART-CNN, 2=FLAN-T5 (default: 0)
  -x, --extractive      Enable extractive QA model for better answer generation
  --enhance-only ENHANCE_ONLY
                        Path to existing JSON file to enhance (skip Pass 1)
```

#### Model Recommendations:

- **For Best Quality**: Use `-m 2 -x` (FLAN-T5 + extractive QA)
- **For Speed**: Use `-m 0` (T5 small)
- **For Balanced Performance**: Use `-m 1 -x` (BART + extractive QA)

#### Two-Pass System:

**Pass 1: Enhanced Generation**
- Content-aware text chunking with mathematical sequence preservation
- Context type classification (definition, sequence, explanation, etc.)
- Multi-model answer generation (extractive + generative)
- Quality scoring for each QA pair

**Pass 2: Enhancement & Validation**
- Advanced quality validation with multiple criteria
- Question type classification and difficulty assessment
- Topic keyword extraction
- Comprehensive filtering and enhancement

#### Usage Examples:

```bash
# Full two-pass generation with best quality settings
python3 genQA.py -i document.pdf -o output.json -m 2 -x -w ./weights

# Enhancement-only mode (improve existing QA pairs)
python3 genQA.py --enhance-only existing_qa.json -o enhanced_output.json

# Speed-optimized generation
python3 genQA.py -i document.pdf -o output.json -m 0 -w ./weights
```

#### Enhanced Output Format:

The output JSON now includes comprehensive metadata:

```json
{
    "qa_pairs": [
        {
            "question": "What is a triangular number?",
            "answer": "A triangular number is formed by adding consecutive natural numbers.",
            "context": "Triangular numbers are 1, 3, 6, 10, 15...",
            "source": "extractive",
            "content_type": "definition",
            "key_concepts": ["triangular", "sequence", "numbers"],
            "model_used": "google/flan-t5-base",
            "quality_score": 0.85,
            "question_type": "definition",
            "difficulty_level": "basic",
            "topic_keywords": ["pattern", "sequence", "number"]
        }
    ],
    "total_pairs": 150,
    "metadata": {
        "model_used": "google/flan-t5-base",
        "extractive_enabled": true,
        "passes_completed": 2,
        "source_file": "math_textbook.pdf"
    },
    "quality_stats": {
        "average_quality_score": 0.78,
        "total_generated": 200,
        "total_enhanced": 150,
        "total_filtered": 50
    }
}
```

### 2. Model Fine-tuning

The fine-tuning script now uses DistilBERT for question-answering with enhanced features:

```bash
python3 fineTuneSLM.py -i input_dir -t tmp_dir -w output_dir [OPTIONS]

Required arguments:
  -i, --input-dir       Directory containing JSON files with QA pairs
  -t, --tmp-dir         Directory for saving training checkpoints
  -w, --output-dir      Directory to save the final fine-tuned model

Optional arguments:
  --resume-from         Path to pre-trained model checkpoint for resume training
  --num-epochs          Number of training epochs (default: 3)
  --batch-size          Batch size for training and evaluation (default: 8)
  --learning-rate       Learning rate for training (default: 2e-5)
```

#### Features:

- **Advanced Dataset Processing**: Intelligent answer position mapping in context
- **Multi-device Support**: Automatic detection of CUDA, MPS, or CPU
- **Resume Training**: Continue from previous checkpoints
- **Evaluation Metrics**: Built-in evaluation with validation split
- **Optimized Training**: FP16 support, gradient clipping, and learning rate scheduling

#### Example:

```bash
# Basic fine-tuning
python3 fineTuneSLM.py -i ./qa_data -t ./checkpoints -w ./fine_tuned_model

# Advanced fine-tuning with custom parameters
python3 fineTuneSLM.py -i ./qa_data -t ./checkpoints -w ./fine_tuned_model \
    --num-epochs 5 --batch-size 16 --learning-rate 3e-5

# Resume training from checkpoint
python3 fineTuneSLM.py -i ./qa_data -t ./checkpoints -w ./fine_tuned_model \
    --resume-from ./previous_model --num-epochs 2
```

### 3. Query Interface

The query interface now supports both CLI and web server modes:

```bash
python3 querySLM.py -m model_dir [OPTIONS]

Required arguments:
  -m, --model_dir       Directory containing the fine-tuned model

Mode selection:
  --mode {cli,server}   Run mode: 'cli' for command-line or 'server' for web server (default: cli)

CLI mode arguments:
  -i, --input           Path to input text file containing questions (one per line)
  -o, --output          Path to output text file for answers

Server mode arguments:
  --port                Port for the web server (default: 5000)
```

#### Examples:

```bash
# CLI mode - batch processing
python3 querySLM.py -m ./fine_tuned_model --mode cli -i questions.txt -o answers.txt

# Web server mode
python3 querySLM.py -m ./fine_tuned_model --mode server --port 8080
```

The web interface provides a simple GUI accessible at `http://localhost:5000` for interactive querying.

## Complete Workflow Example

Here's a comprehensive example workflow:

### 1. Generate Enhanced QA Pairs
```bash
# Download and cache all models first
python3 genQA.py -i textbook.pdf -o qa_pairs.json -m 2 -x -w ./weights
```

### 2. Verify Quality and Enhance if Needed
```bash
# Optional: Run enhancement pass on existing data
python3 genQA.py --enhance-only qa_pairs.json -o enhanced_qa_pairs.json
```

### 3. Fine-tune the Model
```bash
# Create training data directory
mkdir training_data
mv enhanced_qa_pairs.json training_data/

# Fine-tune model
python3 fineTuneSLM.py -i training_data -t ./checkpoints -w ./specialized_model \
    --num-epochs 5 --batch-size 16
```

### 4. Query the Specialized Model
```bash
# Interactive web interface
python3 querySLM.py -m ./specialized_model --mode server

# Or batch processing
echo "What is a prime number?" > test_questions.txt
python3 querySLM.py -m ./specialized_model --mode cli -i test_questions.txt -o answers.txt
```

## Quality Features

### Enhanced Content Processing
- **Text Cleaning**: Advanced encoding fix and noise removal
- **Mathematical Sequences**: Special handling for number patterns and sequences
- **Content Classification**: Automatic categorization (definition, example, sequence, etc.)
- **Smart Chunking**: Context-aware text segmentation

### Quality Validation
- **Question Validation**: Structure, length, and content quality checks
- **Answer Validation**: Relevance, completeness, and noise filtering
- **Quality Scoring**: Multi-criteria scoring system (0.0 to 1.0)
- **Automatic Filtering**: Remove low-quality pairs based on configurable thresholds

### Multi-Model Support
- **T5 Models**: Small QA-QG, FLAN-T5 Base for question generation
- **BART**: Large CNN model for text summarization and QA
- **DistilBERT**: Extractive QA for high-quality answer extraction
- **Hybrid Approach**: Combine extractive and generative methods

## Model Weight Management

The system automatically manages model weights in the specified directory:

```
weights/
├── models--valhalla--t5-small-qa-qg-hl/           # T5 QA-QG model (-m 0)
├── models--facebook--bart-large-cnn/              # BART model (-m 1)
├── models--google--flan-t5-base/                  # FLAN-T5 model (-m 2)
├── models--distilbert-base-uncased-distilled-squad/ # Extractive QA (-x flag)
└── models--distilbert-base-uncased/               # Fine-tuning base model
```

## Performance Optimization

### Hardware Support
- **CUDA**: Automatic GPU acceleration when available
- **MPS**: Apple Silicon optimization for M1/M2 Macs
- **CPU**: Fallback with optimized threading
- **Mixed Precision**: FP16 training support for memory efficiency

### Processing Optimization
- **Batch Processing**: Efficient handling of multiple QA pairs
- **Memory Management**: Smart chunking to handle large documents
- **Caching**: Model weight caching to avoid re-downloads
- **Progress Tracking**: Detailed progress indicators for long operations

## Output Statistics

The enhanced system provides comprehensive statistics:

```
PROCESSING COMPLETE - FINAL SUMMARY
============================================================
Source PDF: textbook.pdf
Model used: google/flan-t5-base
Extractive QA: Enabled
Total characters extracted: 25,340

PASS 1 RESULTS:
  Generated QA pairs: 200

PASS 2 RESULTS:
  Enhanced pairs: 150
  Filtered out: 50
  Final pairs in output: 150
  Average quality score: 0.782

QUALITY BREAKDOWN:
  Answer sources: {'extractive': 105, 'generative': 45}
  Question types: {'definition': 60, 'explanation': 45, 'mathematical_pattern': 30, 'general': 15}
  Difficulty levels: {'basic': 80, 'intermediate': 50, 'advanced': 20}
============================================================
```

## Technical Requirements

### System Requirements
- **Python**: 3.8+ (tested on 3.9+)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large documents)
- **Storage**: 5GB+ for model weights and temporary files
- **GPU**: Optional but recommended (CUDA or MPS support)

### Dependencies
```bash
pip install torch>=1.9.0 transformers>=4.20.0 pdfplumber>=0.7.0 datasets>=2.0.0 
pip install sentencepiece>=0.1.97 scikit-learn>=1.1.0 tqdm>=4.64.0 flask>=2.2.0
```

