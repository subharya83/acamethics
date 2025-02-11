# PDF Question-Answer Generation and Fine-tuning

This project provides a set of tools for generating question-answer pairs from PDF documents and fine-tuning a T5 language model for specialized question-answering tasks. The system consists of three main components:

## Components

1. **PDF QA Generator (genQA.py)**
   - Extracts text from PDF documents
   - Generates question-answer pairs using a pre-trained T5 model
   - Saves the generated QA pairs in JSON format

2. **Model Fine-tuning (fineTuneSLM.py)**
   - Fine-tunes a T5 model using generated QA pairs
   - Supports customizable training parameters
   - Saves the fine-tuned model for later use

3. **Query Interface (querySLM.py)**
   - Loads a fine-tuned model
   - Generates answers for user-provided questions
   - Supports batch processing of questions

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- pdfplumber
- datasets

Install the required packages:
```bash
pip install torch transformers pdfplumber datasets
```

## Usage

### 1. Generate QA Pairs from PDF

```bash
python genQA.py -i input.pdf -o output.json -w weights
```

Arguments:
- `-i, --input`: Path to the input PDF file
- `-o, --output`: Path to save the generated QA pairs (JSON format)
- `-w, --weights`: Directory to store model weights (default: "weights")

### 2. Fine-tune the Model

```bash
python fineTuneSLM.py -d data_dir -o output_dir [-m model_name] [-e epochs] [-b batch_size]
```

Arguments:
- `-d, --data_dir`: Directory containing QA pairs in JSON format
- `-o, --output_dir`: Directory to save the fine-tuned model
- `-m, --model_name`: Pre-trained model to fine-tune (default: "t5-small")
- `-e, --epochs`: Number of training epochs (default: 3)
- `-b, --batch_size`: Training batch size (default: 8)

### 3. Generate Answers Using Fine-tuned Model

```bash
python querySLM.py -i questions.txt -o answers.txt -m model_dir
```

Arguments:
- `-i, --input`: Text file containing questions (one per line)
- `-o, --output`: Output file for generated answers
- `-m, --model_dir`: Directory containing the fine-tuned model

## Example Workflow

1. Generate QA pairs from a PDF:
```bash
python genQA.py -i document.pdf -o qa_pairs.json
```

2. Fine-tune the model using generated QA pairs:
```bash
python fineTuneSLM.py -d ./data -o ./fine_tuned_model -e 5
```

3. Use the fine-tuned model to answer questions:
```bash
python querySLM.py -i my_questions.txt -o answers.txt -m ./fine_tuned_model
```

## Technical Details

- The project uses the T5 model architecture for both QA generation and answering
- GPU acceleration is automatically used when available
- QA generation uses the `valhalla/t5-small-qa-qg-hl` pre-trained model
- Fine-tuning supports mixed precision training on compatible GPUs
- The system handles text chunking to work within model token limits

## Notes

- The quality of generated QA pairs depends on the clarity and structure of the input PDF
- Fine-tuning performance may vary based on the quality and quantity of training data
- GPU availability significantly affects processing speed
- Large PDFs may require significant processing time and memory

## Limitations

- Maximum input length is limited to 512 tokens
- PDF extraction may not preserve complex formatting
- Model performance depends on the quality of training data
- GPU memory requirements increase with batch size