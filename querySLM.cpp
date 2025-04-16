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

// Function to load the model
static llama_model * load_model(const std::string & model_path, llama_context_params & ctx_params) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, model_path.c_str());

    llama_model * model = llama_load_model_from_file(model_path.c_str(), ctx_params);
    if (!model) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, model_path.c_str());
        return nullptr;
    }

    return model;
}

// Function to generate answers
static void generate_answers(
    const std::string & input_file,
    const std::string & output_file,
    llama_model * model,
    llama_context * ctx) {

    // Read questions from input file
    std::ifstream in_file(input_file);
    if (!in_file.is_open()) {
        fprintf(stderr, "Error: Could not open input file %s\n", input_file.c_str());
        return;
    }

    std::vector<std::string> questions;
    std::string line;
    while (std::getline(in_file, line)) {
        if (!line.empty()) {
            questions.push_back(line);
        }
    }
    in_file.close();

    // Prepare output file
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        fprintf(stderr, "Error: Could not open output file %s\n", output_file.c_str());
        return;
    }

    // Generate answers for each question
    for (const auto & question : questions) {
        // Format the prompt (similar to the Python version)
        std::string prompt = "answer question: " + question;

        // Tokenize the prompt
        std::vector<llama_token> tokens = ::llama_tokenize(ctx, prompt, true);

        // Evaluate the tokens
        if (llama_eval(ctx, tokens.data(), tokens.size(), 0, 1)) {
            fprintf(stderr, "Error: Failed to evaluate prompt\n");
            continue;
        }

        // Generate the answer
        const int n_len = 512; // max length
        std::string answer;

        for (int i = 0; i < n_len; i++) {
            // Get the next token
            llama_token id = llama_sample_token(ctx, NULL, NULL);

            if (id == llama_token_eos(model)) {
                break; // Stop if we reach the end-of-sequence token
            }

            // Convert token to string
            std::string token_str = llama_token_to_piece(ctx, id);
            answer += token_str;

            // Evaluate the new token
            if (llama_eval(ctx, &id, 1, tokens.size() + i, 1)) {
                fprintf(stderr, "Error: Failed to evaluate token\n");
                break;
            }
        }

        // Write to output file
        out_file << "Question: " << question << "\n";
        out_file << "Answer: " << answer << "\n\n";
    }

    out_file.close();
    fprintf(stderr, "Answers saved to %s\n", output_file.c_str());
}

int main(int argc, char ** argv) {
    // Parse command line arguments
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    // Initialize llama.cpp
    llama_backend_init(params.numa);

    // Initialize model and context
    llama_model * model;
    llama_context * ctx;
    llama_context_params ctx_params = llama_context_default_params();

    // Load the model
    model = load_model(params.model, ctx_params);
    if (model == nullptr) {
        return 1;
    }

    // Create context
    ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "Error: Failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    // Generate answers
    generate_answers(params.input_path, params.output_path, model, ctx);

    // Cleanup
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
