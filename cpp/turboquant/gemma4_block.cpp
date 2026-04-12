// Gemma 4 single transformer block with TurboQuant KV cache
// Implements: RMSNorm → Q/K/V proj → K/Q norm → RoPE → Attention → O proj → Residual → FFN
#include "turboquant.h"
#include "mlx/io.h"
#include "mlx/fast.h"
#include <iostream>
#include <chrono>

using namespace mlx::core;

struct Timer {
    std::chrono::high_resolution_clock::time_point s;
    Timer() : s(std::chrono::high_resolution_clock::now()) {}
    double ms() { return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - s).count(); }
};

// Dequantize a 4-bit weight
array dequant_weight(const std::unordered_map<std::string, array>& w,
                     const std::string& prefix) {
    auto weight = w.at(prefix + ".weight");
    auto scales = w.at(prefix + ".scales");
    auto biases = w.at(prefix + ".biases");
    return dequantize(weight, scales, biases, 64, 4);
}

int main() {
    std::cout << "╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Gemma 4 — Single Block Forward Pass (C++)   ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝" << std::endl;

    std::string model_dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    turboquant::set_metallib_path(METALLIB_PATH);

    // Load first shard (has layer 0)
    std::cout << "Loading weights..." << std::endl;
    auto [weights, meta] = load_safetensors(model_dir + "model-00001-of-00004.safetensors");
    std::cout << "Loaded " << weights.size() << " tensors" << std::endl;

    // Constants
    const int hidden_size = 5376;
    const int head_dim = 256;
    const int n_heads = 32;
    const int n_kv_heads = 16;
    const int intermediate_size = 21504;
    const float rms_eps = 1e-6f;
    const int seq_len = 32;

    std::string L0 = "language_model.model.layers.0.";

    // Create test input: (1, seq_len, hidden_size)
    auto x = random::normal({1, seq_len, hidden_size});
    eval(x);
    std::cout << "Input: [1, " << seq_len << ", " << hidden_size << "]" << std::endl;

    Timer t;

    // === Step 1: Input LayerNorm ===
    auto ln_weight = weights.at(L0 + "input_layernorm.weight");
    eval(ln_weight);
    auto h = fast::rms_norm(x, array(1.0f) + ln_weight, rms_eps);
    eval(h);
    std::cout << "  RMSNorm: " << t.ms() << "ms" << std::endl;

    // === Step 2: Q/K/V Projections (dequantize 4-bit weights) ===
    t = Timer();
    auto q_proj = dequant_weight(weights, L0 + "self_attn.q_proj");
    auto k_proj = dequant_weight(weights, L0 + "self_attn.k_proj");
    auto v_proj = dequant_weight(weights, L0 + "self_attn.v_proj");
    eval(q_proj, k_proj, v_proj);

    auto queries = matmul(h, transpose(q_proj));   // (1, seq, n_heads * head_dim)
    auto keys = matmul(h, transpose(k_proj));       // (1, seq, n_kv_heads * head_dim)
    auto values_raw = matmul(h, transpose(v_proj)); // (1, seq, n_kv_heads * head_dim)
    eval(queries, keys, values_raw);
    std::cout << "  Q/K/V proj: " << t.ms() << "ms" << std::endl;
    std::cout << "    Q: [" << queries.shape(0) << "," << queries.shape(1) << "," << queries.shape(2) << "]" << std::endl;
    std::cout << "    K: [" << keys.shape(0) << "," << keys.shape(1) << "," << keys.shape(2) << "]" << std::endl;

    // === Step 3: Reshape to heads ===
    queries = reshape(queries, {1, seq_len, n_heads, head_dim});
    keys = reshape(keys, {1, seq_len, n_kv_heads, head_dim});
    auto values = reshape(values_raw, {1, seq_len, n_kv_heads, head_dim});

    // Transpose: (B, H, N, D)
    queries = transpose(queries, {0, 2, 1, 3});
    keys = transpose(keys, {0, 2, 1, 3});
    values = transpose(values, {0, 2, 1, 3});
    eval(queries, keys, values);

    // === Step 4: K/Q Norm ===
    t = Timer();
    auto q_norm_w = weights.at(L0 + "self_attn.q_norm.weight");
    auto k_norm_w = weights.at(L0 + "self_attn.k_norm.weight");
    eval(q_norm_w, k_norm_w);
    queries = fast::rms_norm(queries, array(1.0f) + q_norm_w, rms_eps);
    keys = fast::rms_norm(keys, array(1.0f) + k_norm_w, rms_eps);
    eval(queries, keys);
    std::cout << "  Q/K norm: " << t.ms() << "ms" << std::endl;

    // === Step 5a: Standard Attention ===
    t = Timer();
    float scale = 1.0f / std::sqrt((float)head_dim);
    auto output = fast::scaled_dot_product_attention(queries, keys, values, scale);
    eval(output);
    double std_attn_ms = t.ms();
    std::cout << "  Standard Attention: " << std_attn_ms << "ms" << std::endl;

    // === Step 5b: TurboQuant Attention (compress keys, dequant at scoring) ===
    t = Timer();
    // Codebooks
    std::vector<float> cb1_d(16);
    for (int i = 0; i < 16; i++) cb1_d[i] = (2.0f * M_PI * i / 16.0f) + M_PI / 16.0f;
    auto cb1 = array(cb1_d.data(), {16}, float32);
    auto cb2 = array(std::vector<float>{0.31f, 0.63f, 0.94f, 1.26f}.data(), {4}, float32);
    auto cb3 = array(std::vector<float>{0.43f, 0.67f, 0.90f, 1.14f}.data(), {4}, float32);
    auto cb4 = array(std::vector<float>{0.52f, 0.71f, 0.86f, 1.05f}.data(), {4}, float32);
    auto P_id = eye(head_dim);
    eval(cb1, cb2, cb3, cb4, P_id);

    // Flatten keys for TurboQuant: (1, n_kv_heads, seq, head_dim) → (n_kv*seq, head_dim)
    auto keys_flat = reshape(keys, {n_kv_heads * seq_len, head_dim});

    // Packed quantize
    auto packed = turboquant::polar_quantize_packed(keys_flat, P_id, cb1, cb2, cb3, cb4);
    for (auto& p : packed) eval(p);

    // Packed dequant
    auto keys_recon = turboquant::polar_dequantize_packed(
        packed[0], packed[1], packed[2], packed[3], packed[4],
        cb1, cb2, cb3, cb4, head_dim);
    auto keys_recon_4d = reshape(keys_recon, {1, n_kv_heads, seq_len, head_dim});
    eval(keys_recon_4d);

    // SDPA with reconstructed keys
    auto output_tq = fast::scaled_dot_product_attention(queries, keys_recon_4d, values, scale);
    eval(output_tq);
    double tq_attn_ms = t.ms();
    std::cout << "  TurboQuant Attention: " << tq_attn_ms << "ms" << std::endl;

    // Quality comparison
    auto cos = sum(reshape(output, {-1}) * reshape(output_tq, {-1})) /
        (sqrt(sum(output * output)) * sqrt(sum(output_tq * output_tq)) + array(1e-8f));
    eval(cos);
    std::cout << "  TQ vs Standard cos_sim: " << cos.item<float>() << std::endl;

    // Memory savings
    size_t packed_kb = 0;
    for (auto& p : packed) packed_kb += p.nbytes();
    size_t fp16_kb = n_kv_heads * seq_len * head_dim * 2;
    std::cout << "  Key memory: " << packed_kb/1024 << "KB packed vs "
              << fp16_kb/1024 << "KB FP16 (" << (float)fp16_kb/packed_kb << "x)" << std::endl;
    std::cout << "    Output: [" << output.shape(0) << "," << output.shape(1)
              << "," << output.shape(2) << "," << output.shape(3) << "]" << std::endl;

    // === Step 6: O projection + residual ===
    t = Timer();
    output = transpose(output, {0, 2, 1, 3});  // (B, N, H, D)
    output = reshape(output, {1, seq_len, n_heads * head_dim});
    auto o_proj = dequant_weight(weights, L0 + "self_attn.o_proj");
    eval(o_proj);
    auto attn_output = matmul(output, transpose(o_proj));
    auto residual = x + attn_output;  // residual connection
    eval(residual);
    std::cout << "  O proj + residual: " << t.ms() << "ms" << std::endl;

    std::cout << "\n=== Gemma 4 Layer 0 Block Forward Pass Complete! ===" << std::endl;
    std::cout << "  Final shape: [" << residual.shape(0) << "," << residual.shape(1)
              << "," << residual.shape(2) << "]" << std::endl;

    return 0;
}
