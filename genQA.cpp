#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <ggml/ggml.h>
#include <llama.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Configuration structure
struct Config {
    std::string model_path = "models/t5-small-qa-qg-hl.gguf";
    std::string weights_dir = "weights";
    int max_length = 512;
    int n_threads = 4;
    int n_gpu_layers = 0; // 0 for CPU-only
};

// QA Pair structure
struct QAPair {
    std::string question;
    std::string answer;
};

// PDF text extractor (placeholder - would need actual PDF library)
std::string extract_text_from_pdf(const std::string& pdf_path) {
    // In a real implementation, you would use a PDF library like Poppler
    // This is just a placeholder that reads a text file
    std::ifstream file(pdf_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + pdf_path);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Split text into chunks
std::vector<std::string> split_text(const std::string& text, size_t chunk_size) {
    std::vector<std::string> chunks;
    for (size_t i = 0; i < text.size(); i += chunk_size) {
        chunks.push_back(text.substr(i, chunk_size));
    }
    return chunks;
}

// Load GGUF model
llama_model* load_gguf_model(const Config& config) {
    llama_backend_init();
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    
    llama_model* model = llama_load_model_from_file(config.model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("Failed to load GGUF model: " + config.model_path);
    }
    
    return model;
}

// Generate QA pairs using GGUF model
std::vector<QAPair> generate_qa_pairs(llama_model* model, const std::string& text, const Config& config) {
    std::vector<QAPair> qa_pairs;
    auto chunks = split_text(text, config.max_length);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed = 1234;
    ctx_params.n_ctx = config.max_length;
    ctx_params.n_threads = config.n_threads;
    ctx_params.n_threads_batch = config.n_threads;
    
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        throw std::runtime_error("Failed to create context");
    }
    
    for (const auto& chunk : chunks) {
        std::string prompt = "generate questions: " + chunk;
        
        llama_batch batch = llama_batch_init(512, 0);
        batch.n_tokens = prompt.size();
        
        // Tokenize the prompt (simplified - actual implementation would use proper tokenization)
        for (size_t i = 0; i < prompt.size(); i++) {
            batch.token[i] = prompt[i];
            batch.pos[i] = i;
            batch.seq_id[i] = 0;
            batch.logits[i] = false;
        }
        batch.logits[batch.n_tokens - 1] = true;
        
        // Run inference
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Failed to decode batch" << std::endl;
            continue;
        }
        
        // Generate response
        std::string response;
        int n_cur = batch.n_tokens;
        int n_len = 256; // max response length
        
        while (n_cur <= n_len) {
            auto logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
            auto vocab_size = llama_n_vocab(model);
            
            // Simple greedy decoding (in practice you'd want better sampling)
            llama_token new_token_id = std::distance(logits, std::max_element(logits, logits + vocab_size));
            
            if (new_token_id == llama_token_eos(model)) {
                break;
            }
            
            response += static_cast<char>(new_token_id);
            
            // Prepare next batch
            llama_batch_clear(batch);
            llama_batch_add(batch, new_token_id, n_cur, {0}, false);
            
            if (llama_decode(ctx, batch) != 0) {
                std::cerr << "Failed to decode" << std::endl;
                break;
            }
            
            n_cur++;
        }
        
        // Parse response into QA pairs (simplified)
        QAPair pair;
        size_t q_pos = response.find("Q:");
        size_t a_pos = response.find("A:");
        
        if (q_pos != std::string::npos && a_pos != std::string::npos) {
            pair.question = response.substr(q_pos + 2, a_pos - q_pos - 2);
            pair.answer = response.substr(a_pos + 2);
            qa_pairs.push_back(pair);
        }
        
        llama_batch_free(batch);
    }
    
    llama_free(ctx);
    return qa_pairs;
}

// Save QA pairs to JSON file
void save_qa_pairs(const std::vector<QAPair>& qa_pairs, const std::string& output_file) {
    json j;
    for (const auto& pair : qa_pairs) {
        j.push_back({
            {"question", pair.question},
            {"answer", pair.answer}
        });
    }
    
    std::ofstream out(output_file);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_file);
    }
    
    out << j.dump(4);
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " -i <input_pdf> -o <output_json> [-w <weights_dir>]" << std::endl;
        return 1;
    }
    
    std::string input_pdf, output_json, weights_dir = "weights";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_pdf = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_json = argv[++i];
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            weights_dir = argv[++i];
        }
    }
    
    if (input_pdf.empty() || output_json.empty()) {
        std::cerr << "Both input and output paths are required" << std::endl;
        return 1;
    }
    
    try {
        Config config;
        config.weights_dir = weights_dir;
        
        // Step 1: Load GGUF model
        std::cout << "Loading GGUF model..." << std::endl;
        llama_model* model = load_gguf_model(config);
        std::cout << "Model loaded." << std::endl;
        
        // Step 2: Extract text from PDF
        std::cout << "Extracting text from PDF..." << std::endl;
        std::string text = extract_text_from_pdf(input_pdf);
        std::cout << "Text extracted." << std::endl;
        
        // Step 3: Generate QA pairs
        std::cout << "Generating QA pairs..." << std::endl;
        auto qa_pairs = generate_qa_pairs(model, text, config);
        std::cout << "Generated " << qa_pairs.size() << " QA pairs." << std::endl;
        
        // Step 4: Save QA pairs
        std::cout << "Saving QA pairs to " << output_json << std::endl;
        save_qa_pairs(qa_pairs, output_json);
        
        // Cleanup
        llama_free_model(model);
        llama_backend_free();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
