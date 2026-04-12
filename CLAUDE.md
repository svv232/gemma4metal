# TurboQuant — Gemma 4 31B Inference on Apple Silicon

## CRITICAL: Never Stop

This is an open-ended research and engineering project. There is ALWAYS more to do.

## What We Built (86 experiments)

**Gemma 4 31B running on a single M1 Max MacBook Pro (64GB)** with auto-adaptive KV cache compression, chunked prefill for 4K+ context, and 10-12 tok/s decode.

### Performance
- **10-12 tok/s** decode at short context (int4 mode)
- **5-9 tok/s** at long context (FP16 mode)
- **11 ms/tok** prefill, **1 ms** prompt cache reload
- **Chunked prefill**: enables 4K+ context (tested to 3.8K with needle extraction)
- **256K context**: 47.9 GB total (FP16 KV 30.5 + weights 17.4) — fits 64GB theoretically

### Auto-Adaptive KV Cache
- **≤950 tokens**: int4 quantization (6.4x compression, perfect quality)
- **>950 tokens**: auto-switches to FP16 (2x compression, no quality loss)
- **Runtime safety**: 950-entry eviction prevents compound error in int4 mode
- Sliding window causal mask enables >1024 token prefill

## Key Scientific Findings

1. **PolarQuant incompatible with Gemma4**: attn_scale=1.0 amplifies angular quantization error
2. **QJL correction worsens quality**: +6.5% perplexity (variance > bias in softmax)
3. **V precision > K precision**: V quantization hurts attention output more than K
4. **Compound error**: int4/int8 KV compounds across 60 layers, fails beyond ~950 tokens
5. **FP16 KV works at any length**: the practical solution for long context
6. **Weight matmuls = 77%** of decode time (KV cache ops = 2%)

## Gemma 4 Architecture (fully decoded)

- **60 layers**: 50 sliding_attention (hd=256, nkv=16, window=1024) + 10 full_attention (hd=512, nkv=4)
- **Norms**: `rms_norm(x) * weight` (NOT `1+weight` like Gemma2)
- **Attention scale**: 1.0 (q/k norms handle magnitude)
- **v_norm**: bare RMS norm (no learnable weight, `with_scale=False`)
- **layer_scalar**: multiplies ENTIRE hidden state at end of layer
- **RoPE**: sliding=default(theta=10K), global=proportional(theta=1M, 128/512 dims rotated)
- **Activation**: gelu_pytorch_tanh
- **V=K**: global layers have no v_proj (attention_k_eq_v=True)

## Hardware

- Apple M1 Max, 10-core CPU, 32-core GPU, 64GB unified memory
- Metal 4, 32KB threadgroup memory per threadgroup

## Usage

```bash
# Build
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8

# Interactive chat (tokenizes with Python, runs C++ inference)
python3 chat_repl.py

# Single prompt (writes to prompt_491.bin, runs binary)
python3 chat.py "Your question here"

# Direct binary (reads prompt_491.bin or uses default)
./build/gemma4_multilayer
```

## File Structure

```
turboquant/
├── CLAUDE.md                  ← you are here
├── gemma4_multilayer.cpp      ← main inference (341 lines, auto-adaptive KV)
├── gemma4_multiturn.cpp       ← multi-turn conversation demo
├── gemma4_bench.cpp           ← FP32 vs int4 benchmark
├── turboquant.h               ← Metal kernel API (PolarQuant primitives)
├── turboquant.cpp             ← MLX Primitive implementations
├── turboquant.metal           ← Metal compute shaders
├── chat_repl.py               ← interactive REPL
├── chat.py                    ← single-prompt wrapper
├── vocab.bin                  ← BPE vocabulary for token decoding
├── CMakeLists.txt             ← build system (links MLX from source)
├── polar_quality_sweep.cpp    ← PolarQuant codebook quality sweep
├── kv_precision_sweep.cpp     ← KV bit-width precision analysis
├── context_scaling.cpp        ← decode speed vs context length
└── profile_decode.cpp         ← per-component timing breakdown
```

## KV Cache Compression Pareto Frontier

| Mode | Compression | Max Context | Quality | 256K Memory |
|------|------------|-------------|---------|-------------|
| int4 K+V | 6.4x | ~950 tokens | Perfect | 23.4 GB* |
| FP16 K+V | 2.0x | Any | Perfect | 47.9 GB |
| FP32 K+V | 1.0x | Any | Perfect | 60.9 GB |

*int4 only for ≤950 tokens due to compound error across 60 layers

## Papers
- TurboQuant: arxiv.org/abs/2504.19874 (ICLR 2026)
- QJL: arxiv.org/abs/2406.03482 (AAAI)
- PolarQuant: arxiv.org/abs/2502.02617 (AISTATS 2026)

## What's Next
- Speculative decoding with Gemma 2 2B draft model (2x throughput potential)
- Chunked prefill for >1024 int4 context (needs proper cross-chunk attention)
- BFloat16 computation for reduced intermediate memory
- Weight offloading for models that exceed memory
