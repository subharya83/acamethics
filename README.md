### Acamethics

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

Install the required packages:
```bash
pip install torch transformers pdfplumber datasets sentencepiece
```

## Usage

### 1. Generate QA Pairs from PDF

```
usage: genQA.py [-h] -i INPUT -o OUTPUT [-w WEIGHTS] [-m {0,1,2}] [-x]

Generate QA pairs from a PDF file for SLM fine-tuning.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input PDF file.
  -o OUTPUT, --output OUTPUT
                        Path to the output JSON file.
  -w WEIGHTS, --weights WEIGHTS
                        Directory to store model weights.
  -m {0,1,2}, --model {0,1,2}
                        Model choice: 0=valhalla/t5-small-qa-qg-hl, 1=facebook/bart-large-cnn, 2=google/flan-t5-base
  -x, --extractive      Enable extractive QA model for better answer generation

```
#### Model Recommendations:

- For Best Quality: Use -m 2 -x (FLAN-T5 + extractive QA)
- For Speed: Use -m 0 (T5 small)
- For Balanced Performance: Use -m 1 -x (BART + extractive QA)

#### Sanity check/Unit tests after installation

```bash
# Run once to download each models (-m 0, -m 1, -m 2)
python3 genQA.py -i document.pdf -o output.json -m 2 -x -w ./weights
```

```bash
# Check directory structure to verify if all weights are downloaded
tree weights

weights/
├── models--valhalla--t5-small-qa-qg-hl/        # T5 model (if using -m 0)
├── models--facebook--bart-large-cnn/           # BART model (if using -m 1)  
├── models--google--flan-t5-base/               # FLAN-T5 model (if using -m 2)
└── models--distilbert-base-uncased-distilled-squad/  # Extractive QA model (if using -x)
```

```bash
# Running with input data
python3 genQA.py -i input/CBSE-Class-6-Maths-Chapter-01.pdf -o input/CBSE-Class-6-Maths-Chapter-01.json -m 2 -w weights -x

# Expected results
Selected model: google/flan-t5-base
Extractive QA model: ENABLED
Loading FLAN-T5 Base Model (google/flan-t5-base)...
...
Device set to use cpu
Successfully loaded FLAN-T5 Base Model
Loading extractive QA model...
Device set to use cpu
Successfully loaded extractive QA model
Models loaded successfully.
Extracting text from PDF...
Extracted 13028 characters from PDF.
Processing text and generating QA pairs...


Successfully saved 100 QA pairs to input/CBSE-Class-6-Maths-Chapter-01.json
Process completed! Generated 100 QA pairs.

Generation Summary:
  extractive: 70 pairs
  keyword_extraction: 4 pairs
  generative: 26 pairs
```

### 2. Fine-tune the Model

```bash
python3 fineTuneSLM.py -i input -t tmp -w weights
Using device: mps
Downloading tokenizer and model to weights
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 24.7kB/s]
vocab.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 1.13MB/s]
tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 3.64MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 483/483 [00:00<00:00, 714kB/s]
model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 268M/268M [00:00<00:00, 321MB/s]
Some weights of DistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['qa_outputs.bias', 'qa_outputs.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Loading JSON files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 1380.05it/s]
Loaded 890 QA pairs
Starting fine-tuning...
  0%|                                                                                                                                                                                   | 0/336 [00:00<?, ?it/s]/Users/subhabrata.bhattacharya/Library/Python/3.9/lib/python/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 5.8135, 'grad_norm': 10.624452590942383, 'learning_rate': 9.900000000000002e-06, 'epoch': 0.89}                                                                                                        
{'loss': 3.2865, 'grad_norm': 12.551226615905762, 'learning_rate': 1.9900000000000003e-05, 'epoch': 1.79}                                                                                                       
{'loss': 2.1938, 'grad_norm': 22.015884399414062, 'learning_rate': 2.9900000000000002e-05, 'epoch': 2.68}                                                                                                       
{'train_runtime': 317.6801, 'train_samples_per_second': 8.405, 'train_steps_per_second': 1.058, 'train_loss': 3.5600887026105608, 'epoch': 3.0}                                                                 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 336/336 [05:17<00:00,  1.06it/s]
Saving fine-tuned model to weights
```

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
