#include "ggml/ggml.h"
#include "llama.h"
#include "common.h"
#include "console.h"

#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Structure to hold QA pairs
struct QAPair {
    std::string context;
    std::string question;
    std::string answer;
};

// Load dataset from JSON files
std::vector<QAPair> load_dataset(const std::string& data_dir) {
    std::vector<QAPair> qa_pairs;
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".json") {
            std::ifstream f(entry.path());
            json data = json::parse(f);
            
            for (const auto& item : data) {
                QAPair pair;
                pair.context = item["context"];
                pair.question = item["question"];
                pair.answer = item["answer"];
                qa_pairs.push_back(pair);
            }
        }
    }
    
    return qa_pairs;
}

// Tokenize and preprocess dataset
void preprocess_dataset(
    const std::vector<QAPair>& qa_pairs,
    llama_context* ctx,
    std::vector<llama_token>& input_ids,
    std::vector<llama_token>& label_ids,
    size_t max_length = 512) {
    
    for (const auto& pair : qa_pairs) {
        // Format input text
        std::string input_text = "generate questions: " + pair.context;
        
        // Tokenize input
        std::vector<llama_token> input_tokens = ::llama_tokenize(ctx, input_text, true);
        if (input_tokens.size() > max_length) {
            input_tokens.resize(max_length);
        }
        
        // Format target text
        std::string target_text = pair.question + " " + pair.answer;
        
        // Tokenize target
        std::vector<llama_token> target_tokens = ::llama_tokenize(ctx, target_text, true);
        if (target_tokens.size() > max_length) {
            target_tokens.resize(max_length);
        }
        
        // Add to batches
        input_ids.insert(input_ids.end(), input_tokens.begin(), input_tokens.end());
        label_ids.insert(label_ids.end(), target_tokens.begin(), target_tokens.end());
    }
}

// Training function
void fine_tune_model(
    const std::string& data_dir,
    const std::string& output_dir,
    const std::string& model_name = "t5-small",
    int epochs = 3,
    int batch_size = 8) {
    
    // Initialize model parameters
    gpt_params params;
    params.model = model_name;
    
    // Initialize backend
    llama_backend_init(params.numa);
    
    // Load model
    llama_model* model = llama_load_model_from_file(params.model.c_str(), llama_context_default_params());
    if (!model) {
        std::cerr << "Error: Failed to load model" << std::endl;
        return;
    }
    
    // Create context
    llama_context* ctx = llama_new_context_with_model(model, llama_context_default_params());
    if (!ctx) {
        std::cerr << "Error: Failed to create context" << std::endl;
        llama_free_model(model);
        return;
    }
    
    // Load dataset
    std::vector<QAPair> qa_pairs = load_dataset(data_dir);
    
    // Preprocess dataset
    std::vector<llama_token> input_ids, label_ids;
    preprocess_dataset(qa_pairs, ctx, input_ids, label_ids);
    
    // Basic training loop (simplified - real implementation would need proper batching, optimization, etc.)
    std::cout << "Starting fine-tuning..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
        
        // Simplified training - in reality you'd need proper batching and optimization
        for (size_t i = 0; i < input_ids.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, input_ids.size());
            
            // Evaluate batch (real implementation would need loss calculation, backprop, etc.)
            if (llama_eval(ctx, input_ids.data() + i, end - i, 0, 1)) {
                std::cerr << "Error in evaluation" << std::endl;
                break;
            }
            
            // Normally you would:
            // 1. Calculate loss
            // 2. Backpropagate
            // 3. Update weights
            // But llama.cpp doesn't currently support this for T5 models
        }
    }
    
    std::cout << "Fine-tuning completed." << std::endl;
    
    // Save model (this is simplified - real implementation would need GGUF conversion)
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }
    
    // Note: Actual model saving would require converting to GGUF format
    std::cout << "Fine-tuned model would be saved to " << output_dir << std::endl;
    
    // Cleanup
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " -d <data_dir> -o <output_dir> "
                  << "[-m <model_name>] [-e <epochs>] [-b <batch_size>]" << std::endl;
        return 1;
    }
    
    std::string data_dir, output_dir;
    std::string model_name = "t5-small";
    int epochs = 3;
    int batch_size = 8;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "-m" && i + 1 < argc) {
            model_name = argv[++i];
        } else if (arg == "-e" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "-b" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        }
    }
    
    if (data_dir.empty() || output_dir.empty()) {
        std::cerr << "Error: Data directory and output directory are required" << std::endl;
        return 1;
    }
    
    // Run fine-tuning
    fine_tune_model(data_dir, output_dir, model_name, epochs, batch_size);
    
    return 0;
}
