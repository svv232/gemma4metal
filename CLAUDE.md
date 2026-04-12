# TurboQuant — Fast Gemma 4 on Apple Silicon

## CRITICAL: Never Stop

This is an open-ended research and engineering project. There is ALWAYS more to do.

## Results (167 experiments)

### Best Configuration: MoE 26B-A4B at 59 tok/s
The MoE model with 4B active parameters is **4x faster** than the 31B dense at lower memory:

| Model | tok/s | Memory | Quality |
|-------|-------|--------|---------|
| **26B-A4B MoE** | **59** | **14.5 GB** | **Good** |
| 26B-A4B MoE + int4 KV | 53 | 14.5 GB | Good |
| 31B dense (mlx-lm) | 15 | 17.5 GB | Excellent |
| 31B dense + int4 KV | 14.8 | 17.4 GB | Excellent |
| **31B + TurboQuant fused** | **9.8** | **17.4 GB** | **Excellent** |
| 31B dense (C++ baseline) | 7.2-10.8 | 17.4 GB | Excellent |
| E2B (2B) | 111 | 3.6 GB | Fair |

### Speed Optimizations Exhaustively Explored
| Approach | Result | Why |
|----------|--------|-----|
| MoE model (4B active) | **59 tok/s (+4x)** | Less bandwidth needed |
| Remove CPU top-p sort | +58% (6.4→10.2) | 262K sort was bottleneck |
| int4 KV cache | 6.4x compression | MLX native quantize |
| **Fused sdpa_int4 Metal kernel** | **+35% at 786 tok** | Vectorized 4-bit, no K deq |
| quantized_matmul SDPA (per-head) | 0.87x | Dispatch overhead per head |
| Self-speculative (10 layers) | 0% match | Layer skip too disruptive |
| Spec decode (E2B→31B) | 12 tok/s (slower) | 25% acceptance too low |
| Spec decode (E2B→MoE) | 43 tok/s (slower) | Same issue |
| 2-bit weights | 0.67x | MLX 2-bit kernel slow |

### Key Scientific Findings
1. **MoE > dense for Apple Silicon** — 4B active params at 59 tok/s vs 31B dense at 15
2. **PolarQuant incompatible with Gemma4** — attn_scale=1.0 amplifies angular error
3. **QJL worsens quality** — +6.5% perplexity (variance > bias in softmax)
4. **int4 KV compounds across 60 layers** — fails beyond ~950 tokens
5. **FP16 KV works to 4K** with chunked prefill; BF16 (8-bit mantissa) fails
6. **Speculative decode not viable with E2B** — 25% acceptance (need >60%)
7. **Chat template**: `<|turn>` format (not `<start_of_turn>`)
8. **Fused sdpa_int4 kernel 35% faster** than dequantize+SDPA at 786 tokens
9. **int8 KV also fails** beyond ~1000 tokens (attn_scale=1.0 fundamental limit)
10. **mlx-lm 50% faster than C++** for weight matmuls — mx.compile graph compilation

### Gemma 4 31B Architecture
- 60 layers: 50 sliding (hd=256, nkv=16, w=1024) + 10 global (hd=512, nkv=4)
- Norms: `rms_norm(x) * weight`, attention scale=1.0, bare v_norm
- RoPE: sliding=10K, global=proportional(1M, 128/512 dims)
- gelu_pytorch_tanh, V=K in global layers

## Quick Start
```bash
# Fastest: MoE via mlx-lm (59 tok/s, needs Python 3.12)
/opt/homebrew/bin/python3.12 mlx_moe_chat.py "What is the capital of France?"

# 31B via mlx-lm (15 tok/s, needs Python 3.12)
/opt/homebrew/bin/python3.12 mlx_chat.py "What is the capital of France?"

# 31B via C++ (10 tok/s, custom engine)
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make gemma4 -j8
python3 chat.py "What is the capital of France?"

# Tests (C++ engine)
./run_tests.sh   # 11/11
```

## File Structure
```
turboquant/
├── CLAUDE.md                  ← you are here
├── mlx_moe_chat.py            ← FASTEST: MoE 59 tok/s streaming chat
├── mlx_chat.py                ← 31B via mlx-lm (15 tok/s + int4 KV)
├── gemma4_multilayer.cpp      ← C++ 31B engine (10 tok/s)
├── turboquant.h/.cpp/.metal   ← PolarQuant Metal kernels
├── chat.py / chat_repl.py     ← C++ Python wrappers
├── speculative_decode.py      ← spec decode prototype
├── gemma4_e2b_draft.cpp       ← E2B draft model (WIP)
├── run_tests.sh               ← 11/11 test suite
├── vocab.bin                  ← BPE vocabulary
├── CMakeLists.txt
└── benchmarks/                ← 5 benchmark tools
```

## Papers
- TurboQuant: arxiv.org/abs/2504.19874 (ICLR 2026)
- QJL: arxiv.org/abs/2406.03482 (AAAI)
- PolarQuant: arxiv.org/abs/2502.02617 (AISTATS 2026)
