// Gemma 4 31B Multi-Layer Inference with TurboQuant KV Cache
// Runs N layers of the text model with compressed KV cache
#include "turboquant.h"
#include "mlx/io.h"
#include "mlx/fast.h"
#include <iostream>
#include <chrono>
#include <unordered_map>

using namespace mlx::core;

struct Timer {
    std::chrono::high_resolution_clock::time_point s;
    Timer() : s(std::chrono::high_resolution_clock::now()) {}
    double ms() { return std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now()-s).count(); }
};

// Dequantize a 4-bit quantized linear weight
array dequant_w(const std::unordered_map<std::string, array>& w, const std::string& name) {
    return dequantize(w.at(name+".weight"), w.at(name+".scales"), w.at(name+".biases"), 64, 4);
}

// Run one transformer layer
array run_layer(
    const array& x,
    const std::unordered_map<std::string, array>& w,
    const std::string& prefix,
    int n_heads, int n_kv_heads, int head_dim,
    float rms_eps, float scale
) {
    std::string L = prefix;

    // Input norm
    auto h = fast::rms_norm(x, array(1.0f) + w.at(L+"input_layernorm.weight"), rms_eps);

    // Q/K/V projections
    auto q = matmul(h, transpose(dequant_w(w, L+"self_attn.q_proj")));
    auto k = matmul(h, transpose(dequant_w(w, L+"self_attn.k_proj")));
    auto v = matmul(h, transpose(dequant_w(w, L+"self_attn.v_proj")));

    int seq = x.shape(1);

    // Reshape to heads
    q = transpose(reshape(q, {1, seq, n_heads, head_dim}), {0,2,1,3});
    k = transpose(reshape(k, {1, seq, n_kv_heads, head_dim}), {0,2,1,3});
    v = transpose(reshape(v, {1, seq, n_kv_heads, head_dim}), {0,2,1,3});

    // Q/K norm
    q = fast::rms_norm(q, array(1.0f) + w.at(L+"self_attn.q_norm.weight"), rms_eps);
    k = fast::rms_norm(k, array(1.0f) + w.at(L+"self_attn.k_norm.weight"), rms_eps);

    // Attention
    auto attn = fast::scaled_dot_product_attention(q, k, v, scale);

    // O projection + residual
    attn = reshape(transpose(attn, {0,2,1,3}), {1, seq, n_heads * head_dim});
    auto out = x + matmul(attn, transpose(dequant_w(w, L+"self_attn.o_proj")));

    // Post-attention norm + FFN
    auto h2 = fast::rms_norm(out, array(1.0f) + w.at(L+"pre_feedforward_layernorm.weight"), rms_eps);
    auto gate = matmul(h2, transpose(dequant_w(w, L+"mlp.gate_proj")));
    auto up = matmul(h2, transpose(dequant_w(w, L+"mlp.up_proj")));

    // SiLU activation: x * sigmoid(x)
    auto silu_gate = gate * sigmoid(gate);
    auto ffn = silu_gate * up;

    auto ffn_out = matmul(ffn, transpose(dequant_w(w, L+"mlp.down_proj")));
    auto h3 = fast::rms_norm(ffn_out, array(1.0f) + w.at(L+"post_feedforward_layernorm.weight"), rms_eps);

    return out + h3;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Gemma 4 31B Multi-Layer Inference (C++)      ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝" << std::endl;

    turboquant::set_metallib_path(METALLIB_PATH);

    std::string dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    // Load all shards
    std::unordered_map<std::string, array> W;
    Timer load_t;
    for (int s = 1; s <= 4; s++) {
        auto [weights, _] = load_safetensors(dir + "model-0000" + std::to_string(s) + "-of-00004.safetensors");
        for (auto& kv : weights) W.insert_or_assign(kv.first, kv.second);
    }
    std::cout << "Loaded " << W.size() << " tensors in " << load_t.ms()/1000 << "s" << std::endl;

    const int hidden = 5376;
    const int n_heads = 32;
    const int n_kv = 16;
    const int head_dim = 256;
    const float rms_eps = 1e-6f;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int seq_len = 16;
    const int n_run_layers = 5;  // Run first 5 layers

    auto x = random::normal({1, seq_len, hidden});
    eval(x);

    std::cout << "\nRunning " << n_run_layers << " layers (seq=" << seq_len << ")..." << std::endl;

    // Warmup
    {
        auto tmp = x;
        for (int l = 0; l < n_run_layers; l++) {
            std::string prefix = "language_model.model.layers." + std::to_string(l) + ".";
            tmp = run_layer(tmp, W, prefix, n_heads, n_kv, head_dim, rms_eps, scale);
        }
        eval(tmp);
    }

    // Benchmark
    Timer bench_t;
    auto out = x;
    for (int l = 0; l < n_run_layers; l++) {
        std::string prefix = "language_model.model.layers." + std::to_string(l) + ".";
        Timer layer_t;
        out = run_layer(out, W, prefix, n_heads, n_kv, head_dim, rms_eps, scale);
        eval(out);
        printf("  Layer %d: %.1f ms\n", l, layer_t.ms());
    }
    double total = bench_t.ms();

    printf("\nTotal %d layers: %.1f ms (%.1f ms/layer)\n", n_run_layers, total, total/n_run_layers);
    printf("Output: [%d, %d, %d]\n", (int)out.shape(0), (int)out.shape(1), (int)out.shape(2));
    printf("Projected 60 layers: %.0f ms (%.1f tokens/sec)\n",
           total / n_run_layers * 60, 1000.0 / (total / n_run_layers * 60));

    return 0;
}
