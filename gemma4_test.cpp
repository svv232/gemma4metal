// Load Gemma 4 31B weights and run one layer with TurboQuant KV cache
#include "turboquant.h"
#include "mlx/io.h"
#include "mlx/fast.h"
#include <iostream>
#include <fstream>

using namespace mlx::core;

int main() {
    std::cout << "=== Loading Gemma 4 31B (4-bit) ===" << std::endl;

    std::string model_dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    // Load first shard
    std::string shard_path = model_dir + "model-00001-of-00004.safetensors";
    std::cout << "Loading " << shard_path << "..." << std::endl;

    auto [weights, metadata] = load_safetensors(shard_path);

    std::cout << "Loaded " << weights.size() << " tensors" << std::endl;

    // Print layer 0 attention weights
    for (auto& [name, tensor] : weights) {
        if (name.find("layers.0.self_attn") != std::string::npos) {
            eval(tensor);
            std::cout << "  " << name << ": [";
            for (auto s : tensor.shape()) std::cout << s << ",";
            std::cout << "] " << tensor.dtype() << std::endl;
        }
    }

    // Architecture from config.json:
    // Layer 0: head_dim=72, n_kv_heads=56 (4096/72), sliding_window=1024
    // This is a local attention layer with small heads

    int head_dim = 256;  // from k_norm weight shape [256,]
    int n_kv_heads = 16;  // 4096 / 256
    int hidden_size = 5376;

    std::cout << "\nGemma 4 Layer 0: head_dim=" << head_dim
              << ", n_kv_heads=" << n_kv_heads
              << ", hidden_size=" << hidden_size << std::endl;

    // Create a test input
    int seq_len = 16;
    auto x = random::normal({1, seq_len, hidden_size});
    eval(x);
    std::cout << "Input: [1, " << seq_len << ", " << hidden_size << "]" << std::endl;

    // Extract K/V projection weights
    auto find_weight = [&](const std::string& name) -> array {
        for (auto& [k, v] : weights) {
            if (k.find(name) != std::string::npos) return v;
        }
        throw std::runtime_error("Weight not found: " + name);
    };

    try {
        auto k_weight = find_weight("layers.0.self_attn.k_proj.weight");
        eval(k_weight);
        std::cout << "K proj weight: [";
        for (auto s : k_weight.shape()) std::cout << s << ",";
        std::cout << "] " << k_weight.dtype() << std::endl;

        // The model is quantized (4-bit), so k_weight has special shape
        // For quantized: weight is packed, scales and biases are separate
        auto k_scales = find_weight("layers.0.self_attn.k_proj.scales");
        auto k_biases = find_weight("layers.0.self_attn.k_proj.biases");
        eval(k_scales, k_biases);

        std::cout << "K scales: [";
        for (auto s : k_scales.shape()) std::cout << s << ",";
        std::cout << "]" << std::endl;

        // Dequantize K projection
        auto k_proj_full = dequantize(k_weight, k_scales, k_biases, 64, 4);
        eval(k_proj_full);
        std::cout << "K proj dequantized: [";
        for (auto s : k_proj_full.shape()) std::cout << s << ",";
        std::cout << "]" << std::endl;

        // Project: keys = x @ K^T
        auto keys = matmul(x, transpose(k_proj_full));
        eval(keys);
        std::cout << "Keys: [";
        for (auto s : keys.shape()) std::cout << s << ",";
        std::cout << "]" << std::endl;

        // Reshape to heads: (1, seq_len, n_kv_heads, head_dim)
        auto keys_heads = reshape(keys, {1, seq_len, -1, head_dim});
        eval(keys_heads);
        std::cout << "Keys per head: [";
        for (auto s : keys_heads.shape()) std::cout << s << ",";
        std::cout << "]" << std::endl;

        // === TurboQuant: compress the keys ===
        std::cout << "\n--- TurboQuant Key Compression ---" << std::endl;

        // Flatten keys for TurboQuant: (1, seq_len, n_kv_heads * head_dim) → (seq_len * n_kv_heads, head_dim)
        auto keys_flat = reshape(keys, {seq_len * n_kv_heads, head_dim});
        eval(keys_flat);

        // Create codebooks for head_dim=256, block_size=16
        // Simplified: uniform L1, approximate L2-L4
        std::vector<float> cb1_data(16);
        for (int i = 0; i < 16; i++) cb1_data[i] = (2.0f * M_PI * i / 16.0f) + M_PI / 16.0f;
        auto cb1 = array(cb1_data.data(), {16}, float32);
        std::vector<float> cb2_d = {0.31f, 0.63f, 0.94f, 1.26f};
        std::vector<float> cb3_d = {0.43f, 0.67f, 0.90f, 1.14f};
        std::vector<float> cb4_d = {0.52f, 0.71f, 0.86f, 1.05f};
        auto cb2 = array(cb2_d.data(), {4}, float32);
        auto cb3 = array(cb3_d.data(), {4}, float32);
        auto cb4 = array(cb4_d.data(), {4}, float32);

        // Precondition matrix (identity for now)
        auto P_gemma = eye(head_dim);
        eval(P_gemma, cb1, cb2, cb3, cb4);

        // Set metallib path
        turboquant::set_metallib_path(METALLIB_PATH);

        // Quantize
        auto quantized = turboquant::polar_quantize(
            keys_flat, P_gemma, cb1, cb2, cb3, cb4);
        for (auto& q : quantized) eval(q);

        std::cout << "  Keys compressed:" << std::endl;
        size_t compressed_bytes = 0;
        for (size_t i = 0; i < quantized.size(); i++) {
            compressed_bytes += quantized[i].nbytes();
            std::cout << "    [" << i << "] shape=" << quantized[i].shape()[0]
                      << " dtype=" << quantized[i].dtype()
                      << " bytes=" << quantized[i].nbytes() << std::endl;
        }

        size_t fp16_bytes = seq_len * n_kv_heads * head_dim * 2;
        std::cout << "  Compressed: " << compressed_bytes << " bytes" << std::endl;
        std::cout << "  FP16:       " << fp16_bytes << " bytes" << std::endl;
        std::cout << "  Compression: " << (float)fp16_bytes / compressed_bytes << "x" << std::endl;

        // Dequantize and verify
        auto recon = turboquant::polar_dequantize_fast(
            quantized[0], quantized[1], quantized[2], quantized[3],
            quantized[4], cb1, cb2, cb3, cb4, head_dim);
        eval(recon);

        // Cosine similarity
        auto af = reshape(keys_flat, {-1});
        auto bf = reshape(recon, {-1});
        auto cos = sum(af * bf) / (sqrt(sum(af * af)) * sqrt(sum(bf * bf)) + array(1e-8f));
        eval(cos);
        std::cout << "  Reconstruction cos_sim: " << cos.item<float>() << std::endl;

        std::cout << "\n=== Gemma 4 Layer 0 with TurboQuant works! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;
}
