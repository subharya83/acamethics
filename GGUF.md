## Full Compilation Instructions

```markdown
## GGUF Architecture/llama.cpp Integration - Full Project Setup

### 1. Install Dependencies

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential cmake git libpoppler-cpp-dev nlohmann-json3-dev
```

#### macOS (using Homebrew)
```bash
brew update
brew install cmake git poppler nlohmann-json
```

### 2. Clone and Build GGML/LLaMA.cpp
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake ..
make -j4

# Copy built libraries to your project
mkdir -p /path/to/lib
cp libggml.a libllama.a /path/to/lib/

# Copy headers
mkdir -p /path/to/your_project/include/ggml
cp ../ggml.h ../ggml-alloc.h ../ggml-backend.h ../ggml-cuda.h ../ggml-metal.h /path/to/include/ggml/
cp ../llama.h /path/to/include/
```

### 3. Project Structure
```
acamethics/
├── CMakeLists.txt
├── include/
│   ├── ggml/          # GGML headers from llama.cpp
│   └── llama.h        # LLaMA header from llama.cpp
├── lib/               # GGML libraries from llama.cpp build
│   ├── libggml.a
│   └── libllama.a
├── models/            # GGUF model files
│   └── t5-small-qa-qg-hl.gguf
├── src/
│   ├── fineTuneSLM.cpp
│   ├── genQA.cpp
│   └── querySLM.cpp
└── README.md
```

### 4. Build the Project
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 5. Run the Programs

#### Fine-tuning the model
```bash
./fineTuneSLM -d /path/to/dataset -o /path/to/output [-m model_name] [-e epochs] [-b batch_size]
```

#### Generating QA pairs from PDF
```bash
./genQA -i input.pdf -o output.json [-w weights_dir]
```

#### Querying the model
```bash
./querySLM -m models/t5-small-qa-qg-hl.gguf --input_path questions.txt --output_path answers.txt
```

### Additional Notes:

1. **Model Files**: Place GGUF model files in the `models/` directory before building.

2. **Cross-Platform**:
   - Windows: Use Visual Studio with C++17 support and adjust paths in CMakeLists.txt
   - macOS/Linux: Follow standard build instructions

3. **Troubleshooting**:
   - Linker errors: Verify library paths in CMakeLists.txt
   - Missing Poppler: Install `libpoppler-cpp-dev` (Linux) or `poppler` (macOS)
   - GPU support: Set `n_gpu_layers` in code and have CUDA/ROCm installed

4. **Optimization**:
   - For best performance, build in Release mode (`-DCMAKE_BUILD_TYPE=Release`)
   - Enable GPU layers if available by adjusting `n_gpu_layers` in the code

5. **Dependencies**:
   - Required: llama.cpp (ggml), nlohmann/json
   - Optional: Poppler (only for genQA)
```
