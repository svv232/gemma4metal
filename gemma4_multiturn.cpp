// Gemma 4 31B Multi-Turn Conversation with int4 KV Cache
// Tests KV cache accumulation across 3 dialogue turns
#include "turboquant.h"
#include "mlx/io.h"
#include "mlx/fast.h"
#include <iostream>
#include <fstream>
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

std::vector<std::string> load_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint32_t size; f.read((char*)&size, 4);
    std::vector<std::string> vocab(size);
    for (uint32_t i = 0; i < size; i++) {
        uint16_t len; f.read((char*)&len, 2);
        vocab[i].resize(len);
        f.read(vocab[i].data(), len);
    }
    return vocab;
}

array bare_rms_norm(const array& x, float eps) {
    return x * rsqrt(mean(x * x, -1, true) + array(eps));
}

int main() {
    turboquant::set_metallib_path(METALLIB_PATH);

    const int hidden_size = 5376;
    const int n_layers = 60;
    const float rms_eps = 1e-6f;
    const float final_softcap = 30.0f;
    const int sliding_window = 1024;
    auto is_global = [](int l) { return (l % 6) == 5; };

    std::vector<float> gf(64);
    for (int i = 0; i < 64; i++) gf[i] = 1.0f / std::pow(1000000.0f, 2.0f * i / 512.0f);
    auto global_rope_freqs = array(gf.data(), {64}, float32);
    eval(global_rope_freqs);

    std::string model_dir = "/Users/andromeda/.cache/huggingface/hub/"
        "models--mlx-community--gemma-4-31b-it-4bit/snapshots/"
        "535c5606372deb5d5ab7e29280f111ef2a8e084e/";

    auto vocab = load_vocab("vocab.bin");
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

    auto embed_w = W.at("language_model.model.embed_tokens.weight");
    auto embed_s = W.at("language_model.model.embed_tokens.scales");
    auto embed_b = W.at("language_model.model.embed_tokens.biases");
    auto final_norm = W.at("language_model.model.norm.weight");
    auto full_embed = dequantize(embed_w, embed_s, embed_b, 64, 4);
    eval(full_embed, final_norm);

    // int4 KV cache
    struct QC {
        std::optional<array> kq, ks, kb, vq, vs, vb;
    };
    std::vector<QC> kv_cache(n_layers);
    int cache_offset = 0;

    // Forward: one layer
    auto run_layer = [&](const array& x, int l) -> array {
        std::string L = "language_model.model.layers." + std::to_string(l) + ".";
        bool global = is_global(l);
        int hd = W.at(L+"self_attn.k_norm.weight").shape(0);
        int kv_dim = W.at(L+"self_attn.k_proj.weight").shape(0);
        int nkv = kv_dim / hd;
        int nh = W.at(L+"self_attn.q_proj.weight").shape(0) / hd;
        int seq = x.shape(1);

        auto h = fast::rms_norm(x, W.at(L+"input_layernorm.weight"), rms_eps);
        auto q = qmm(h, L+"self_attn.q_proj");
        auto k_new = qmm(h, L+"self_attn.k_proj");
        auto v_new = has(L+"self_attn.v_proj.weight") ? qmm(h, L+"self_attn.v_proj") : k_new;

        q = transpose(reshape(q, {1,seq,nh,hd}), {0,2,1,3});
        k_new = transpose(reshape(k_new, {1,seq,nkv,hd}), {0,2,1,3});
        v_new = transpose(reshape(v_new, {1,seq,nkv,hd}), {0,2,1,3});

        q = fast::rms_norm(q, W.at(L+"self_attn.q_norm.weight"), rms_eps);
        k_new = fast::rms_norm(k_new, W.at(L+"self_attn.k_norm.weight"), rms_eps);
        v_new = bare_rms_norm(v_new, rms_eps);

        if (global) {
            q = fast::rope(q, 128, false, std::nullopt, 1.0f, cache_offset, global_rope_freqs);
            k_new = fast::rope(k_new, 128, false, std::nullopt, 1.0f, cache_offset, global_rope_freqs);
        } else {
            q = fast::rope(q, hd, false, 10000.0f, 1.0f, cache_offset);
            k_new = fast::rope(k_new, hd, false, 10000.0f, 1.0f, cache_offset);
        }

        // int4 KV cache
        auto kqv = quantize(k_new, 64, 4);
        auto vqv = quantize(v_new, 64, 4);
        if (kv_cache[l].kq.has_value()) {
            kv_cache[l].kq = concatenate({kv_cache[l].kq.value(), kqv[0]}, 2);
            kv_cache[l].ks = concatenate({kv_cache[l].ks.value(), kqv[1]}, 2);
            kv_cache[l].kb = concatenate({kv_cache[l].kb.value(), kqv[2]}, 2);
            kv_cache[l].vq = concatenate({kv_cache[l].vq.value(), vqv[0]}, 2);
            kv_cache[l].vs = concatenate({kv_cache[l].vs.value(), vqv[1]}, 2);
            kv_cache[l].vb = concatenate({kv_cache[l].vb.value(), vqv[2]}, 2);
        } else {
            kv_cache[l].kq = kqv[0]; kv_cache[l].ks = kqv[1]; kv_cache[l].kb = kqv[2];
            kv_cache[l].vq = vqv[0]; kv_cache[l].vs = vqv[1]; kv_cache[l].vb = vqv[2];
        }

        int total = kv_cache[l].kq.value().shape(2);
        if (!global && total > sliding_window) {
            int s = total - sliding_window;
            auto tr = [&](std::optional<array>& a) { a = slice(a.value(), {0,0,s,0}, a.value().shape()); };
            tr(kv_cache[l].kq); tr(kv_cache[l].ks); tr(kv_cache[l].kb);
            tr(kv_cache[l].vq); tr(kv_cache[l].vs); tr(kv_cache[l].vb);
        }

        auto k_full = dequantize(kv_cache[l].kq.value(), kv_cache[l].ks.value(), kv_cache[l].kb.value(), 64, 4);
        auto v_full = dequantize(kv_cache[l].vq.value(), kv_cache[l].vs.value(), kv_cache[l].vb.value(), 64, 4);

        auto attn = fast::scaled_dot_product_attention(q, k_full, v_full, 1.0f, "causal");
        attn = reshape(transpose(attn, {0,2,1,3}), {1, seq, nh * hd});
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
        return hidden;
    };

    // Generate tokens until EOS, return generated token IDs
    auto generate = [&](int max_tokens) -> std::vector<int> {
        std::vector<int> gen;
        auto h = fast::rms_norm(
            reshape(array(0.0f), {1, 1, hidden_size}),  // dummy, will be replaced
            final_norm, rms_eps);

        for (int step = 0; step < max_tokens; step++) {
            int next_id;
            if (step == 0) {
                // First token comes from prefill logits (already computed)
                return gen;  // Should not reach here
            }
            gen.push_back(next_id);
        }
        return gen;
    };

    // Load multi-turn prompts
    std::ifstream mf("multiturn.bin", std::ios::binary);
    if (!mf) { printf("ERROR: multiturn.bin not found\n"); return 1; }
    uint32_t n_turns; mf.read((char*)&n_turns, 4);
    std::vector<std::vector<int>> turns(n_turns);
    for (uint32_t t = 0; t < n_turns; t++) {
        uint32_t n; mf.read((char*)&n, 4);
        turns[t].resize(n);
        mf.read((char*)turns[t].data(), n * 4);
    }

    printf("\n=== Multi-Turn Conversation Test ===\n");
    printf("Turns: %u\n", n_turns);

    for (uint32_t turn = 0; turn < n_turns; turn++) {
        auto& tokens = turns[turn];

        // Print turn prompt
        printf("\n--- Turn %d (%zu tokens) ---\n", turn+1, tokens.size());
        printf("USER: ");
        for (int t : tokens) {
            if (t < (int)vocab.size()) printf("%s", vocab[t].c_str());
        }
        printf("\n");

        // Prefill this turn's tokens
        Timer t;
        auto pidx = array(tokens.data(), {(int)tokens.size()}, int32);
        auto emb = reshape(take(full_embed, pidx, 0), {1, (int)tokens.size(), hidden_size});
        emb = emb * array(std::sqrt((float)hidden_size));
        auto h = emb;
        for (int l = 0; l < n_layers; l++) h = run_layer(h, l);
        eval(h);
        cache_offset += tokens.size();

        // Get logits from last position
        auto last = slice(h, {0, (int)tokens.size()-1, 0}, {1, (int)tokens.size(), hidden_size});
        last = fast::rms_norm(last, final_norm, rms_eps);
        auto logits = quantized_matmul(last, embed_w, embed_s,
            std::optional<array>(embed_b), true, 64, 4);
        logits = array(final_softcap) * tanh(logits / array(final_softcap));
        eval(logits);

        printf("Prefill: %.1fs\n", t.ms()/1000);

        // Generate response
        printf("MODEL: ");
        auto flat = reshape(logits, {-1});
        int next_id = argmax(flat, 0).item<int32_t>();
        std::vector<int> gen_ids;

        for (int step = 0; step < 150; step++) {
            gen_ids.push_back(next_id);
            if (next_id == 1 || next_id == 106 || next_id == 50) {
                printf(" [EOS]\n"); break;
            }
            if (next_id < (int)vocab.size()) printf("%s", vocab[next_id].c_str());
            fflush(stdout);

            auto idx = array({next_id}, {1}, int32);
            auto e = reshape(take(full_embed, idx, 0), {1, 1, hidden_size});
            e = e * array(std::sqrt((float)hidden_size));
            h = e;
            for (int l = 0; l < n_layers; l++) h = run_layer(h, l);
            eval(h);
            cache_offset++;

            h = fast::rms_norm(h, final_norm, rms_eps);
            logits = quantized_matmul(h, embed_w, embed_s,
                std::optional<array>(embed_b), true, 64, 4);
            logits = array(final_softcap) * tanh(logits / array(final_softcap));
            eval(logits);
            flat = reshape(logits, {-1});
            next_id = argmax(flat, 0).item<int32_t>();
        }
        printf("(%zu tokens generated)\n", gen_ids.size());
    }

    // Memory stats
    size_t cache_bytes = 0;
    for (int l = 0; l < n_layers; l++) {
        if (kv_cache[l].kq.has_value()) {
            eval(kv_cache[l].kq.value());
            cache_bytes += kv_cache[l].kq.value().nbytes() + kv_cache[l].ks.value().nbytes() + kv_cache[l].kb.value().nbytes();
            cache_bytes += kv_cache[l].vq.value().nbytes() + kv_cache[l].vs.value().nbytes() + kv_cache[l].vb.value().nbytes();
        }
    }
    printf("\nTotal context: %d tokens, KV cache: %.1f MB (int4, 6.4x)\n", cache_offset, cache_bytes/1e6);

    return 0;
}
