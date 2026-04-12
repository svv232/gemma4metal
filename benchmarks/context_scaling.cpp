// Context-Length Scaling: measure decode tok/s at various KV cache sizes
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
    const int hidden_size = 5376, n_layers = 60;
    const int sliding_window = 1024;
    const float rms_eps = 1e-6f, attn_scale = 1.0f;
    auto is_global = [](int l) { return (l % 6) == 5; };

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
        auto wit = W.find(key + ".weight");
        if (wit == W.end()) return input;
        std::optional<array> biases = W.at(key + ".biases");
        return quantized_matmul(input, wit->second, W.at(key+".scales"), biases, true, 64, 4);
    };

    printf("\n%-10s %-12s %-12s %-12s\n", "Context", "Decode ms", "tok/s", "KV MB");
    printf("%-10s %-12s %-12s %-12s\n", "-------", "---------", "-----", "-----");

    int context_sizes[] = {1, 50, 200, 500, 800, 1024};

    for (int ctx : context_sizes) {
        // Create KV cache with `ctx` tokens of random data (int4 quantized)
        struct QC {
            array kq = array(0.0f), ks = array(0.0f), kb = array(0.0f);
            array vq = array(0.0f), vs = array(0.0f), vb = array(0.0f);
        };
        std::vector<QC> cache(n_layers);

        for (int l = 0; l < n_layers; l++) {
            int hd = is_global(l) ? 512 : 256;
            int nkv = is_global(l) ? 4 : 16;
            int seq = std::min(ctx, is_global(l) ? ctx : sliding_window);
            auto k = random::normal({1, nkv, seq, hd});
            auto v = random::normal({1, nkv, seq, hd});
            auto kqv = quantize(k, 64, 4);
            auto vqv = quantize(v, 64, 4);
            cache[l] = {kqv[0], kqv[1], kqv[2], vqv[0], vqv[1], vqv[2]};
        }
        // Eval all cache
        for (auto& c : cache) eval(c.kq, c.ks, c.kb, c.vq, c.vs, c.vb);

        // Run N decode steps and measure
        int n_steps = 5;
        auto x = random::normal({1, 1, hidden_size});
        eval(x);

        // Warmup
        for (int l = 0; l < n_layers; l++) {
            std::string L = "language_model.model.layers." + std::to_string(l) + ".";
            bool global = is_global(l);
            int hd = W.at(L+"self_attn.k_norm.weight").shape(0);
            int nkv = W.at(L+"self_attn.k_proj.weight").shape(0) / hd;
            int nh = W.at(L+"self_attn.q_proj.weight").shape(0) / hd;

            auto h = fast::rms_norm(x, W.at(L+"input_layernorm.weight"), rms_eps);
            auto q = qmm(h, L+"self_attn.q_proj");
            auto k_new = qmm(h, L+"self_attn.k_proj");
            auto v_new = has(L+"self_attn.v_proj.weight") ? qmm(h, L+"self_attn.v_proj") : k_new;

            q = transpose(reshape(q, {1,1,nh,hd}), {0,2,1,3});
            k_new = transpose(reshape(k_new, {1,1,nkv,hd}), {0,2,1,3});
            v_new = transpose(reshape(v_new, {1,1,nkv,hd}), {0,2,1,3});

            q = fast::rms_norm(q, W.at(L+"self_attn.q_norm.weight"), rms_eps);
            k_new = fast::rms_norm(k_new, W.at(L+"self_attn.k_norm.weight"), rms_eps);
            v_new = bare_rms_norm(v_new, rms_eps);
            if (global) {
                q = fast::rope(q, 128, false, std::nullopt, 1.0f, ctx, global_rope_freqs);
                k_new = fast::rope(k_new, 128, false, std::nullopt, 1.0f, ctx, global_rope_freqs);
            } else {
                q = fast::rope(q, hd, false, 10000.0f, 1.0f, ctx);
                k_new = fast::rope(k_new, hd, false, 10000.0f, 1.0f, ctx);
            }

            // Append to cache and dequant
            auto k_full = dequantize(cache[l].kq, cache[l].ks, cache[l].kb, 64, 4);
            auto v_full = dequantize(cache[l].vq, cache[l].vs, cache[l].vb, 64, 4);

            auto attn = fast::scaled_dot_product_attention(q, k_full, v_full, attn_scale, "causal");
            attn = reshape(transpose(attn, {0,2,1,3}), {1, 1, nh * hd});
            auto attn_out = qmm(attn, L+"self_attn.o_proj");
            if (has(L+"post_attention_layernorm.weight"))
                attn_out = fast::rms_norm(attn_out, W.at(L+"post_attention_layernorm.weight"), rms_eps);
            auto hidden = x + attn_out;

            if (has(L+"pre_feedforward_layernorm.weight")) {
                auto h2 = fast::rms_norm(hidden, W.at(L+"pre_feedforward_layernorm.weight"), rms_eps);
                auto gate = qmm(h2, L+"mlp.gate_proj");
                auto up = qmm(h2, L+"mlp.up_proj");
                auto gelu = gate * array(0.5f) * (array(1.0f) + tanh(
                    array(0.7978845608028654f) * (gate + array(0.044715f) * power(gate, array(3)))));
                auto ffn = qmm(gelu * up, L+"mlp.down_proj");
                if (has(L+"post_feedforward_layernorm.weight"))
                    ffn = fast::rms_norm(ffn, W.at(L+"post_feedforward_layernorm.weight"), rms_eps);
                hidden = hidden + ffn;
            }
            if (has(L+"layer_scalar")) hidden = hidden * W.at(L+"layer_scalar");
            x = hidden;
        }
        eval(x);

        // Timed decode steps
        Timer t;
        for (int step = 0; step < n_steps; step++) {
            for (int l = 0; l < n_layers; l++) {
                std::string L = "language_model.model.layers." + std::to_string(l) + ".";
                bool global = is_global(l);
                int hd = W.at(L+"self_attn.k_norm.weight").shape(0);
                int nkv = W.at(L+"self_attn.k_proj.weight").shape(0) / hd;
                int nh = W.at(L+"self_attn.q_proj.weight").shape(0) / hd;

                auto h = fast::rms_norm(x, W.at(L+"input_layernorm.weight"), rms_eps);
                auto q = qmm(h, L+"self_attn.q_proj");
                auto k_new = qmm(h, L+"self_attn.k_proj");
                auto v_new = has(L+"self_attn.v_proj.weight") ? qmm(h, L+"self_attn.v_proj") : k_new;
                q = transpose(reshape(q, {1,1,nh,hd}), {0,2,1,3});
                k_new = transpose(reshape(k_new, {1,1,nkv,hd}), {0,2,1,3});
                v_new = transpose(reshape(v_new, {1,1,nkv,hd}), {0,2,1,3});
                q = fast::rms_norm(q, W.at(L+"self_attn.q_norm.weight"), rms_eps);
                k_new = fast::rms_norm(k_new, W.at(L+"self_attn.k_norm.weight"), rms_eps);
                v_new = bare_rms_norm(v_new, rms_eps);
                if (global) {
                    q = fast::rope(q, 128, false, std::nullopt, 1.0f, ctx+step, global_rope_freqs);
                    k_new = fast::rope(k_new, 128, false, std::nullopt, 1.0f, ctx+step, global_rope_freqs);
                } else {
                    q = fast::rope(q, hd, false, 10000.0f, 1.0f, ctx+step);
                    k_new = fast::rope(k_new, hd, false, 10000.0f, 1.0f, ctx+step);
                }
                auto k_full = dequantize(cache[l].kq, cache[l].ks, cache[l].kb, 64, 4);
                auto v_full = dequantize(cache[l].vq, cache[l].vs, cache[l].vb, 64, 4);
                auto attn = fast::scaled_dot_product_attention(q, k_full, v_full, attn_scale, "causal");
                attn = reshape(transpose(attn, {0,2,1,3}), {1, 1, nh * hd});
                auto attn_out = qmm(attn, L+"self_attn.o_proj");
                if (has(L+"post_attention_layernorm.weight"))
                    attn_out = fast::rms_norm(attn_out, W.at(L+"post_attention_layernorm.weight"), rms_eps);
                auto hidden = x + attn_out;
                if (has(L+"pre_feedforward_layernorm.weight")) {
                    auto h2 = fast::rms_norm(hidden, W.at(L+"pre_feedforward_layernorm.weight"), rms_eps);
                    auto gate = qmm(h2, L+"mlp.gate_proj");
                    auto up = qmm(h2, L+"mlp.up_proj");
                    auto gelu = gate * array(0.5f) * (array(1.0f) + tanh(
                        array(0.7978845608028654f) * (gate + array(0.044715f) * power(gate, array(3)))));
                    auto ffn = qmm(gelu * up, L+"mlp.down_proj");
                    if (has(L+"post_feedforward_layernorm.weight"))
                        ffn = fast::rms_norm(ffn, W.at(L+"post_feedforward_layernorm.weight"), rms_eps);
                    hidden = hidden + ffn;
                }
                if (has(L+"layer_scalar")) hidden = hidden * W.at(L+"layer_scalar");
                x = hidden;
            }
            eval(x);
        }
        double total_ms = t.ms();
        double per_tok = total_ms / n_steps;
        double tok_s = 1000.0 / per_tok;

        // KV cache memory
        size_t kv_bytes = 0;
        for (auto& c : cache)
            kv_bytes += c.kq.nbytes() + c.ks.nbytes() + c.kb.nbytes() +
                        c.vq.nbytes() + c.vs.nbytes() + c.vb.nbytes();

        printf("%-10d %-12.1f %-12.1f %-12.1f\n", ctx, per_tok, tok_s, kv_bytes / 1e6);
    }

    return 0;
}
