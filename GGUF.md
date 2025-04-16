## GGUF architecture/llama.cpp integration

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
The code depends on the GGML library (from LLaMA.cpp project):

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake ..
make -j4
```

This will build the necessary GGML libraries. Take note of the build directory location.

### 3. Prepare Your Project Structure

Create a project directory with this structure:
```
your_project/
├── include/           # Header files
│   ├── ggml/          # From llama.cpp
│   └── llama.h        # From llama.cpp
├── src/               # Your source files
│   └── main.cpp       # The code you provided
├── lib/               # GGML libraries
│   └── (from llama.cpp build)
└── CMakeLists.txt     # Build configuration
```

### 4. Create a CMakeLists.txt

Here's a basic CMake configuration:

```cmake
cmake_minimum_required(VERSION 3.15)
project(pdf_qa_gguf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/ggml
)

# Link directories
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(POPPLER REQUIRED poppler-cpp)

# Source files
add_executable(pdf_qa
    src/main.cpp
)

# Link libraries
target_link_libraries(pdf_qa
    ggml
    llama
    ${POPPLER_LIBRARIES}
    stdc++fs  # For filesystem support
)

# Copy GGUF model file after build
add_custom_command(TARGET pdf_qa POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/models/t5-small-qa-qg-hl.gguf
    $<TARGET_FILE_DIR:pdf_qa>/models/
)
```

### 5. Build the Project

```bash
mkdir build
cd build
cmake ..
make -j4
```

### 6. Run the Program

After successful compilation, you can run it with:

```bash
./pdf_qa -i input.pdf -o output.json
```

### Additional Notes:

1. **Model File**: You'll need to place the GGUF model file (`t5-small-qa-qg-hl.gguf`) in the `models/` directory of your project.

2. **Cross-Platform**: For Windows, you'll need to:
   - Use Visual Studio with C++17 support
   - Build GGML/LLaMA.cpp using CMake
   - Adjust the CMakeLists.txt for Windows paths

3. **Troubleshooting**:
   - If you get linker errors, verify all library paths are correct
   - For Poppler issues, ensure you have the C++ bindings installed (`libpoppler-cpp-dev` on Linux)
   - If using GPU layers, you'll need CUDA/ROCm installed

4. **Optimization**: For better performance, you may want to add compiler optimizations to your CMakeLists.txt:
   ```cmake
   if(CMAKE_BUILD_TYPE STREQUAL "Release")
       add_compile_options(-O3 -march=native)
   endif()
   ```

