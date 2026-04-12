// Profile Gemma 4 31B decode: measure time per component
#include "turboquant.h"
#include "mlx/io.h"
#include "mlx/fast.h"
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <cmath>

using namespace mlx::core;

struct Timer {
    std::chrono::high_resolution_clock::time_point s;
    Timer() : s(std::chrono::high_resolution_clock::now()) {}
    double ms() { return std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now()-s).count(); }
    void reset() { s = std::chrono::high_resolution_clock::now(); }
};

array bare_rms_norm(const array& x, float eps) {
    return x * rsqrt(mean(x * x, -1, true) + array(eps));
}

int main() {
    turboquant::set_metallib_path(METALLIB_PATH);

    const int hidden_size = 5376;
    const int n_layers = 60;
    const float rms_eps = 1e-6f;

    std::vector<float> gf(64);
    for (int i = 0; i < 64; i++) gf[i] = 1.0f / std::pow(1000000.0f, 2.0f * i / 512.0f);
    auto global_rope_freqs = array(gf.data(), {64}, float32);
    eval(global_rope_freqs);

    std::string model_dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    std::unordered_map<std::string, array> W;
    for (int s = 1; s <= 4; s++) {
        auto [weights, _] = load_safetensors(
            model_dir + "model-0000" + std::to_string(s) + "-of-00004.safetensors");
        for (auto& kv : weights) W.insert_or_assign(kv.first, kv.second);
    }
    printf("Loaded %zu tensors\n", W.size());

    auto has = [&](const std::string& k) { return W.find(k) != W.end(); };
    auto qmm = [&](const array& input, const std::string& key) -> array {
        std::optional<array> biases = W.at(key + ".biases");
        return quantized_matmul(input, W.at(key+".weight"), W.at(key+".scales"), biases, true, 64, 4);
    };
    auto is_global = [](int l) { return (l % 6) == 5; };

    // Create dummy KV cache (100 tokens)
    struct QC {
        array kq = array(0.0f), ks = array(0.0f), kb = array(0.0f);
        array vq = array(0.0f), vs = array(0.0f), vb = array(0.0f);
    };
    std::vector<QC> cache(n_layers);
    for (int l = 0; l < n_layers; l++) {
        int hd = is_global(l) ? 512 : 256;
        int nkv = is_global(l) ? 4 : 16;
        auto k = random::normal({1, nkv, 100, hd});
        auto v = random::normal({1, nkv, 100, hd});
        auto kqv = quantize(k, 64, 4);
        auto vqv = quantize(v, 64, 4);
        cache[l] = {kqv[0], kqv[1], kqv[2], vqv[0], vqv[1], vqv[2]};
        eval(cache[l].kq, cache[l].ks, cache[l].kb, cache[l].vq, cache[l].vs, cache[l].vb);
    }

    // Profile one decode step
    auto x = random::normal({1, 1, hidden_size});
    eval(x);

    double t_proj = 0, t_attn = 0, t_ffn = 0, t_kv = 0, t_norm = 0;
    int n_profile = 3;  // Average over N runs

    for (int run = 0; run < n_profile; run++) {
        auto h = x;
        for (int l = 0; l < n_layers; l++) {
            std::string L = "language_model.model.layers." + std::to_string(l) + ".";
            bool global = is_global(l);
            int hd = W.at(L+"self_attn.k_norm.weight").shape(0);
            int kv_dim = W.at(L+"self_attn.k_proj.weight").shape(0);
            int nkv = kv_dim / hd;
            int nh = W.at(L+"self_attn.q_proj.weight").shape(0) / hd;

            // Norms
            Timer t;
            auto normed = fast::rms_norm(h, W.at(L+"input_layernorm.weight"), rms_eps);
            eval(normed);
            t_norm += t.ms();

            // Q/K/V projections
            t.reset();
            auto q = qmm(normed, L+"self_attn.q_proj");
            auto k_new = qmm(normed, L+"self_attn.k_proj");
            auto v_new = has(L+"self_attn.v_proj.weight") ? qmm(normed, L+"self_attn.v_proj") : k_new;
            eval(q, k_new, v_new);
            t_proj += t.ms();

            // Reshape + norms + RoPE
            t.reset();
            q = transpose(reshape(q, {1,1,nh,hd}), {0,2,1,3});
            k_new = transpose(reshape(k_new, {1,1,nkv,hd}), {0,2,1,3});
            v_new = transpose(reshape(v_new, {1,1,nkv,hd}), {0,2,1,3});
            q = fast::rms_norm(q, W.at(L+"self_attn.q_norm.weight"), rms_eps);
            k_new = fast::rms_norm(k_new, W.at(L+"self_attn.k_norm.weight"), rms_eps);
            v_new = bare_rms_norm(v_new, rms_eps);
            if (global) {
                q = fast::rope(q, 128, false, std::nullopt, 1.0f, 100, global_rope_freqs);
                k_new = fast::rope(k_new, 128, false, std::nullopt, 1.0f, 100, global_rope_freqs);
            } else {
                q = fast::rope(q, 256, false, 10000.0f, 1.0f, 100);
                k_new = fast::rope(k_new, 256, false, 10000.0f, 1.0f, 100);
            }
            eval(q, k_new, v_new);
            t_norm += t.ms();

            // KV cache: quantize new + dequant full
            t.reset();
            auto kqv = quantize(k_new, 64, 4);
            auto vqv = quantize(v_new, 64, 4);
            auto k_full = dequantize(cache[l].kq, cache[l].ks, cache[l].kb, 64, 4);
            auto v_full = dequantize(cache[l].vq, cache[l].vs, cache[l].vb, 64, 4);
            eval(k_full, v_full);
            t_kv += t.ms();

            // Attention
            t.reset();
            auto attn = fast::scaled_dot_product_attention(q, k_full, v_full, 1.0f, "causal");
            attn = reshape(transpose(attn, {0,2,1,3}), {1, 1, nh * hd});
            auto attn_out = qmm(attn, L+"self_attn.o_proj");
            if (has(L+"post_attention_layernorm.weight"))
                attn_out = fast::rms_norm(attn_out, W.at(L+"post_attention_layernorm.weight"), rms_eps);
            eval(attn_out);
            t_attn += t.ms();

            auto hidden = h + attn_out;

            // FFN
            if (has(L+"pre_feedforward_layernorm.weight")) {
                t.reset();
                auto h2 = fast::rms_norm(hidden, W.at(L+"pre_feedforward_layernorm.weight"), rms_eps);
                auto gate = qmm(h2, L+"mlp.gate_proj");
                auto up = qmm(h2, L+"mlp.up_proj");
                auto gelu = gate * array(0.5f) * (array(1.0f) + tanh(
                    array(0.7978845608028654f) * (gate + array(0.044715f) * power(gate, array(3)))));
                auto ffn = qmm(gelu * up, L+"mlp.down_proj");
                if (has(L+"post_feedforward_layernorm.weight"))
                    ffn = fast::rms_norm(ffn, W.at(L+"post_feedforward_layernorm.weight"), rms_eps);
                eval(ffn);
                t_ffn += t.ms();
                hidden = hidden + ffn;
            }
            if (has(L+"layer_scalar")) hidden = hidden * W.at(L+"layer_scalar");
            eval(hidden);
            h = hidden;
        }
    }

    double total = t_proj + t_attn + t_ffn + t_kv + t_norm;
    printf("\n=== Decode Profile (avg of %d runs, 100-token context) ===\n", n_profile);
    printf("  QKV+O Projections: %6.1f ms (%4.1f%%)\n", t_proj/n_profile, 100*t_proj/total);
    printf("  Attention (SDPA):  %6.1f ms (%4.1f%%)\n", t_attn/n_profile, 100*t_attn/total);
    printf("  FFN (gate+up+down):%6.1f ms (%4.1f%%)\n", t_ffn/n_profile, 100*t_ffn/total);
    printf("  KV cache (q+dq):   %6.1f ms (%4.1f%%)\n", t_kv/n_profile, 100*t_kv/total);
    printf("  Norms+RoPE:        %6.1f ms (%4.1f%%)\n", t_norm/n_profile, 100*t_norm/total);
    printf("  TOTAL:             %6.1f ms (%.1f tok/s)\n", total/n_profile, 1000.0*n_profile/total);

    return 0;
}
