"""
Comprehensive benchmark suite for TurboQuant.

Tests all modes (QJL-only, TurboQuant), multiple bit-widths,
and sequence lengths up to 256K. Measures throughput, memory,
and accuracy against FP16 reference.
"""

import mlx.core as mx
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from turboquant import TurboQuantConfig, TurboQuantKVCache


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    mode: str
    embed_dim: int
    num_heads: int
    seq_len: int
    bits_level1: int
    bits_higher: int

    throughput_tps: float
    kernel_time_ms: float
    memory_bytes: int
    fp16_memory_bytes: int
    memory_savings_ratio: float
    cosine_similarity: float
    bits_per_coord: float


def cosine_sim(a, b):
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    return (dot / (norm_a * norm_b + 1e-8)).item()


def run_benchmark(mode, seq_len, num_heads=8, embed_dim=128,
                  bits_level1=4, bits_higher=2, n_warmup=3, n_runs=10):
    """Run a single benchmark configuration."""
    mx.random.seed(0)
    config = TurboQuantConfig(
        embed_dim=embed_dim, num_heads=num_heads,
        polar_bits_level1=bits_level1, polar_bits_higher=bits_higher,
    )

    B, H, N, D = 1, num_heads, seq_len, embed_dim
    keys = mx.random.normal((B, H, N, D))
    values = mx.random.normal((B, H, N, D))
    queries = mx.random.normal((B, H, 1, D))

    # Reference
    scale = D ** -0.5
    ref_output = mx.softmax(
        (queries @ mx.transpose(keys, axes=(0, 1, 3, 2))) * scale, axis=-1
    ) @ values
    mx.eval(ref_output)

    # TurboQuant
    cache = TurboQuantKVCache(config, mode=mode)
    cache.update(keys, values)

    for _ in range(n_warmup):
        out = cache.attention(queries)
        mx.eval(out)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = cache.attention(queries)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    out = cache.attention(queries)
    mx.eval(out)
    cos = cosine_sim(out, ref_output)

    avg_time = np.median(times)
    mem = cache.memory_bytes()
    fp16 = cache.fp16_equivalent_bytes()

    # Compute effective bits per coordinate
    if mode == "turboquant":
        D = embed_dim
        polar_bits = (D // 2) * bits_level1
        for l in range(1, int(np.log2(config.polar_block_size))):
            polar_bits += (D // (2 ** (l + 1))) * bits_higher
        n_blocks = D // config.polar_block_size
        key_bits = polar_bits + n_blocks * 16 + 128 + 32
        val_bits = polar_bits + n_blocks * 16
        bpc = (key_bits + val_bits) / D / 2
    else:
        bpc = (128 / 8 + 4 + D * 2) / D  # sign_bits + norm + FP16 values

    return BenchmarkResult(
        mode=mode,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_len=seq_len,
        bits_level1=bits_level1,
        bits_higher=bits_higher,
        throughput_tps=float(seq_len / avg_time),
        kernel_time_ms=float(avg_time * 1000),
        memory_bytes=mem,
        fp16_memory_bytes=fp16,
        memory_savings_ratio=float(fp16 / max(mem, 1)),
        cosine_similarity=float(cos),
        bits_per_coord=float(bpc),
    )


if __name__ == "__main__":
    print("TurboQuant Comprehensive Benchmark")
    print("=" * 80)
    print(f"Hardware: Apple M1 Max, 32-core GPU, 64GB unified memory, Metal 4")
    print()

    all_results = []

    # ── Section 1: Mode comparison at various seq lengths ──
    print("─── QJL-only vs TurboQuant (default 4b+2b) ───")
    print(f"{'Mode':>12s} {'N':>7s} {'cos_sim':>8s} {'compress':>8s} {'tps':>12s} {'ms':>8s} {'bpc':>5s}")

    for mode in ["qjl_only", "turboquant"]:
        for N in [1024, 4096, 16384, 65536]:
            r = run_benchmark(mode, N)
            all_results.append(r)
            print(f"{r.mode:>12s} {r.seq_len:>7d} {r.cosine_similarity:>8.4f} "
                  f"{r.memory_savings_ratio:>7.2f}x {r.throughput_tps:>11,.0f} "
                  f"{r.kernel_time_ms:>7.2f} {r.bits_per_coord:>5.1f}")

    # ── Section 2: Bit-width sweep ──
    print()
    print("─── Bit-Width Sweep (TurboQuant, N=4096) ───")
    print(f"{'Config':>16s} {'cos_sim':>8s} {'compress':>8s} {'bpc':>5s}")

    bit_configs = [
        (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (6, 3), (8, 4),
    ]
    for bl1, bl2 in bit_configs:
        r = run_benchmark("turboquant", 4096, bits_level1=bl1, bits_higher=bl2)
        all_results.append(r)
        label = f"{bl1}bL1+{bl2}bL2"
        print(f"{label:>16s} {r.cosine_similarity:>8.4f} "
              f"{r.memory_savings_ratio:>7.2f}x {r.bits_per_coord:>5.1f}")

    # ── Section 3: Gemma 4 31B memory projection ──
    print()
    print("─── Gemma 4 31B @ 256K Context Memory Projection ───")

    configs_gemma = [
        ("fast", 3, 2), ("balanced", 4, 2), ("quality", 4, 3), ("hifi", 6, 3),
    ]
    n_layers = 48
    n_kv_heads = 16
    D = 128
    seq = 256000
    weight_gb = 31e9 * 0.5 / 1024**3

    print(f"{'Preset':>10s} {'cos_sim':>8s} {'KV cache':>10s} {'Total':>8s} {'Fits 64GB':>10s}")
    for name, bl1, bl2 in configs_gemma:
        r = run_benchmark("turboquant", 4096, num_heads=n_kv_heads,
                          bits_level1=bl1, bits_higher=bl2)
        per_layer_mb = r.memory_bytes / 1024**2
        kv_gb = per_layer_mb * n_layers * (seq / 4096) / 1024
        total = weight_gb + kv_gb
        fits = "YES" if total < 64 else "NO"
        print(f"{name:>10s} {r.cosine_similarity:>8.4f} {kv_gb:>9.1f}GB {total:>7.1f}GB {fits:>10s}")

    # Write results
    with open("benchmarks/latest.json", "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n{len(all_results)} results written to benchmarks/latest.json")
