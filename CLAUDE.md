# TurboQuant — Metal Kernel Implementation

## CRITICAL: Never Stop

This is an open-ended research and engineering project. There is ALWAYS more to do. After every experiment, immediately start the next one. Do not stop after publishing results. Do not conclude that the project is "done" or "complete." The loop is infinite by design.

If you just finished an experiment, here is what is still ahead — pick the next thing and keep going:
- Accuracy not at 0.99+ cosine similarity? Improve it.
- Metal kernels not benchmarked against MLX reference? Benchmark them.
- Only tested at small sequence lengths? Test at 64K, 128K, 256K.
- Throughput not measured? Measure it.
- Metal kernel not using SIMD reductions or threadgroup memory? Optimize it.
- PolarQuant codebooks not validated against analytical PDFs? Validate them.
- Value compression not as good as key compression? Improve it.
- No end-to-end MLX-LM integration yet? Build it.
- No Gemma 4 31B inference test yet? Set it up.
- Everything working? Run ablations, try different bit-widths, write a benchmark suite.

There is no finish line. Keep the loop running.

## Goal

Implement Google's TurboQuant (QJL + PolarQuant) as Metal compute shaders on Apple Silicon, integrated with MLX, to enable **Gemma 4 31B at full 256K context on a single M1 Max MacBook Pro (64GB)**.

This has never been done. TurboQuant was published March 2026 with no code release (only QJL has CUDA reference code). No Metal implementation exists. Google's own compression research applied to Google's own model on Apple Silicon — a first-of-its-kind implementation.

## Why this matters

Gemma 4 31B at BF16 needs 58.3GB for weights alone — impossible with any meaningful context. Even at 4-bit (17.4GB weights), the FP16 KV cache at 256K context is massive.

Without TurboQuant: 31B at 4-bit weights (17.4GB) + FP16 KV cache at 256K tokens (~30GB) = 47.4GB. Fits, but barely — no room for activations/overhead.
With TurboQuant: 31B at 4-bit weights (17.4GB) + 3-bit KV cache at 256K tokens (~6GB) = 23.4GB. Fits easily with room to spare.

The real unlock: **full 256K context window on a laptop** — a quarter-million tokens of context on consumer hardware. This is Google's own model + Google's own compression, implemented for the first time on Apple Silicon.

## Hardware

- Apple M1 Max, 10-core CPU, 32-core GPU, 64GB unified memory
- Metal 4, 32KB threadgroup memory per threadgroup
- No CUDA — everything must be Metal compute shaders or use MLX/Accelerate

## Architecture

TurboQuant is a two-stage quantizer:
1. **Stage 1 — PolarQuant (b-1 bits):** Random rotation → recursive polar transform → angle quantization per level
2. **Stage 2 — QJL (1 bit):** Compute residual from Stage 1 → random projection → sign bits → unbiased error correction

For KV cache compression at inference:
- **Quantize** key/value vectors as they're produced (prefill + each decode step)
- **Score** by computing attention logits between full-precision queries and quantized keys
- **Dequantize** values for the weighted sum (or use asymmetric estimation)

## Implementation Order

### Phase 1: QJL (has CUDA reference code)
1. `qjl_quant.metal` — project key vectors through random matrix, take sign bits, pack into uint8
2. `qjl_score.metal` — asymmetric inner product: full-precision query sketch × packed sign bits
3. Python wrapper using MLX custom Metal kernels

### Phase 2: PolarQuant (paper-only, no reference code)
4. `polar_transform.metal` — forward polar transform (recursive pairs → angles + radius)
5. `polar_inverse.metal` — inverse polar transform (radius + quantized angles → reconstructed vector)
6. Codebook computation from analytical angle distributions
7. Python wrapper

### Phase 3: TurboQuant Combined
8. Combine: PolarQuant (b-1 bits) + QJL (1 bit residual correction)
9. Full attention kernel: quantized KV cache → attention scores → weighted value sum

### Phase 4: MLX Integration
10. Hook into MLX's KV cache mechanism for Gemma 4 31B
11. End-to-end inference: load 4-bit weights + TurboQuant KV cache → 256K context generation

## Key Technical Details

### QJL Kernel Architecture (port from CUDA)
- Random projection matrix S ∈ R^(sketch_dim × emb_dim), generated via QR of Gaussian
- Quantization: sign(S × k) packed 8 bits per uint8
- Score: q @ S^T gives query sketch, then dot with unpacked sign bits, scale by sqrt(π/2)/dim × ||k||
- Outlier channels handled separately (top-k by norm)

### PolarQuant Algorithm
- Precondition: y = P × x where P is random orthogonal (or randomized Hadamard for O(d log d))
- Forward: pair coordinates, compute (angle, radius) recursively for log2(d) levels
- Level 1 angles ∈ [0, 2π) uniform; level l≥2 angles ∈ [0, π/2] with known PDF concentrating at π/4
- Quantize angles with level-specific codebooks (Lloyd-Max on analytical PDF)
- Store: angle indices (variable bit-width per level) + final radius (FP16)

### Codebook Values
- 1-bit: ±sqrt(2/(πd))
- 2-bit: ±0.453/sqrt(d), ±1.51/sqrt(d)
- 3-bit, 4-bit: must be computed via Lloyd-Max on f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)

## Papers
- TurboQuant: arxiv.org/abs/2504.19874 (ICLR 2026)
- QJL: arxiv.org/abs/2406.03482 (AAAI)
- PolarQuant: arxiv.org/abs/2502.02617 (AISTATS 2026)
- QJL CUDA reference: github.com/amirzandieh/QJL

## What "experiments" mean in this project

This is kernel engineering, not ML model training. An "experiment" is:
- Implementing or optimizing a kernel variant
- Benchmarking: throughput (tokens/sec), memory usage, numerical accuracy vs FP16 reference
- The "metric" is a composite: correctness (max abs error vs FP16) × throughput × memory savings

## Metric

Primary: `composite_score = throughput_tps * memory_savings_ratio * (1 - clamp(max_abs_error / tolerance, 0, 1))`
Where tolerance = 0.01 for attention scores.

Sub-metrics:
- `throughput_tps`: tokens per second for attention computation
- `memory_savings_ratio`: FP16_kv_size / turboquant_kv_size
- `max_abs_error`: maximum absolute error vs FP16 reference attention scores
- `mean_abs_error`: mean absolute error
- `kernel_time_ms`: raw kernel execution time

## File structure

```
turboquant/
├── CLAUDE.md              ← you are here
├── reference/             ← algorithm specs from papers
│   ├── turboquant.md
│   ├── qjl.md
│   └── polarquant.md
├── kernels/metal/         ← Metal compute shaders
├── python/                ← Python/MLX integration
├── benchmarks/            ← performance measurement
└── tests/                 ← correctness validation
```
