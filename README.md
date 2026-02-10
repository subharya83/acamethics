# Acamethics

A toolkit for generating question-answer pairs from middle and high school math
textbook PDFs and fine-tuning a small language model to answer those questions.
The system has three components that form a complete pipeline: generate, train,
and query.

## Components

### 1. QA Generator (`genQA.py`)

Two-pass system that reads a PDF textbook, extracts text, generates QA pairs
using a T5-family model, and then validates and enhances them.

- Content-aware chunking with mathematical sequence preservation
- Extractive + generative answer strategies
- Quality scoring, deduplication, and automatic filtering
- Two model options (T5-QA-QG and FLAN-T5)

### 2. Fine-tuning (`fineTuneSLM.py`)

Fine-tunes a DistilBERT extractive QA model on the generated QA pairs.

- Filters out samples whose answers cannot be mapped to token positions
- Evaluation via the SQuAD metric (F1 / Exact Match)
- Supports checkpoint resume, gradient accumulation, and FP16

### 3. Query Interface (`querySLM.py`)

Loads the fine-tuned DistilBERT model for inference in CLI or web-server mode.

- Batched inference for file-based queries
- Minimal web GUI with a `/health` endpoint
- REST API at `/query`

## Installation

```bash
pip install torch transformers pdfplumber evaluate sentencepiece \
            scikit-learn tqdm flask numpy
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=1.9.0
transformers>=4.20.0
pdfplumber>=0.7.0
evaluate>=0.4.0
sentencepiece>=0.1.97
scikit-learn>=1.1.0
tqdm>=4.64.0
flask>=2.2.0
numpy>=1.22.0
```

## Usage

### 1. Generate QA Pairs from a PDF

```
usage: genQA.py [-h] -i INPUT -o OUTPUT [-w WEIGHTS] [-m {0,1}] [-x]
                [--enhance-only ENHANCE_ONLY]

Two-pass QA pair generator with enhancement

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input PDF file
  -o OUTPUT, --output OUTPUT
                        Path for output JSON file
  -w WEIGHTS, --weights WEIGHTS
                        Directory to store model weights (default: weights)
  -m {0,1}, --model {0,1}
                        Model choice: 0=T5-QA-QG (default), 1=FLAN-T5
  -x, --extractive      Enable extractive QA model for better answer generation
  --enhance-only ENHANCE_ONLY
                        Path to existing JSON file to enhance (skip Pass 1)
```

#### Model Recommendations

- **Best quality:** `-m 1 -x` (FLAN-T5 + extractive QA)
- **Speed:** `-m 0` (T5 small, no extractive)
- **Balanced:** `-m 0 -x` (T5 small + extractive QA)

> The original BART-CNN option (`-m 1` in older versions) was a
> summarization model that could not generate questions. It has been
> replaced by FLAN-T5.

#### Examples

```bash
# Full two-pass generation with best quality
python3 genQA.py -i textbook.pdf -o output.json -m 1 -x -w ./weights

# Enhancement-only mode (improve an existing JSON)
python3 genQA.py -i dummy -o enhanced.json --enhance-only existing_qa.json

# Fast generation
python3 genQA.py -i textbook.pdf -o output.json -m 0 -w ./weights
```

#### Two-Pass System

**Pass 1 -- Generation**
1. Extract and clean text from the PDF.
2. Split into overlapping content-aware chunks.
3. Classify each chunk (definition, sequence, explanation, etc.).
4. Generate questions with the T5 model.
5. Extract or generate answers (extractive preferred).
6. Deduplicate near-identical QA pairs.

**Pass 2 -- Enhancement**
1. Validate question and answer quality.
2. Score each pair on multiple criteria.
3. Filter out low-quality pairs (score < 0.4).
4. Tag with question type, difficulty level, and topic keywords.

#### Output Format

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
    "metadata": { ... },
    "quality_stats": { ... }
}
```

### 2. Fine-tune the Model

```
usage: fineTuneSLM.py [-h] -i INPUT_DIR -t TMP_DIR -w OUTPUT_DIR
                      [--resume-from RESUME_FROM] [--num-epochs NUM_EPOCHS]
                      [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                      [--gradient-accumulation-steps STEPS]
```

The fine-tuning script trains a **DistilBERT** extractive QA model. Samples
whose answers cannot be located inside their context are automatically skipped.

```bash
# Basic fine-tuning
python3 fineTuneSLM.py -i ./qa_data -t ./checkpoints -w ./fine_tuned_model

# Custom hyperparameters with gradient accumulation
python3 fineTuneSLM.py -i ./qa_data -t ./checkpoints -w ./fine_tuned_model \
    --num-epochs 5 --batch-size 16 --learning-rate 3e-5 \
    --gradient-accumulation-steps 2

# Resume from a previous checkpoint
python3 fineTuneSLM.py -i ./qa_data -t ./checkpoints -w ./fine_tuned_model \
    --resume-from ./previous_model --num-epochs 2
```

The input directory should contain one or more JSON files in the same format
produced by `genQA.py`.

### 3. Query the Model

```
usage: querySLM.py [-h] -m MODEL_DIR [--mode {cli,server}]
                   [-i INPUT] [-o OUTPUT] [--port PORT]
```

The query script loads the **DistilBERT** model saved by `fineTuneSLM.py`.
Because it is an extractive QA model, you must provide both a question and a
context passage for best results.

#### CLI Mode

The input file should contain one entry per line. Each line is either a plain
question or a question and context separated by a tab character:

```
What is the sum of angles in a triangle?	The sum of angles in a triangle is 180 degrees.
```

```bash
python3 querySLM.py -m ./fine_tuned_model --mode cli -i questions.txt -o answers.txt
```

#### Web Server Mode

```bash
python3 querySLM.py -m ./fine_tuned_model --mode server --port 8080
```

The web interface is available at `http://localhost:8080` and accepts both a
question and a context passage. A REST endpoint is available at `/query`:

```bash
curl -X POST http://localhost:8080/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is a prime number?", "context": "A prime number has exactly two factors: 1 and itself."}'
```

A health check endpoint is available at `/health`.

## Complete Workflow

```bash
# 1. Generate QA pairs from a textbook PDF
python3 genQA.py -i math_textbook.pdf -o qa_pairs.json -m 1 -x -w ./weights

# 2. (Optional) Run enhancement on existing data
python3 genQA.py -i dummy -o enhanced.json --enhance-only qa_pairs.json

# 3. Fine-tune the model
mkdir -p training_data
cp enhanced.json training_data/
python3 fineTuneSLM.py -i training_data -t ./checkpoints -w ./math_model \
    --num-epochs 5 --batch-size 16

# 4a. Interactive web interface
python3 querySLM.py -m ./math_model --mode server

# 4b. Batch processing
python3 querySLM.py -m ./math_model --mode cli -i questions.txt -o answers.txt
```

## Architecture Notes

All three scripts now use the same model family consistently:

| Stage | Model | Type |
|---|---|---|
| QA Generation (genQA) | T5-QA-QG or FLAN-T5 | Text-to-text (generative) |
| Answer Extraction (genQA, -x) | DistilBERT-SQuAD | Extractive QA |
| Fine-tuning (fineTuneSLM) | DistilBERT | Extractive QA |
| Inference (querySLM) | DistilBERT | Extractive QA |

The generation step uses a T5 model to create questions and optionally a
DistilBERT model to extract answers. The fine-tuning and query steps both
operate on DistilBERT, so saved weights are fully compatible.

## Technical Requirements

- **Python:** 3.8+
- **Memory:** 8 GB RAM minimum, 16 GB+ recommended for large PDFs
- **Storage:** ~5 GB for model weights
- **GPU:** Optional but recommended (CUDA or Apple MPS supported)