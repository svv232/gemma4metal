"""
Benchmark harness for TurboQuant kernels.

Measures throughput, memory savings, and numerical accuracy
against FP16 reference attention computation.
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
    # Config
    embed_dim: int
    num_heads: int
    seq_len: int
    batch_size: int

    # Performance
    throughput_tps: float          # tokens per second (attention computation)
    kernel_time_ms: float          # raw kernel execution time
    memory_bytes: int              # compressed cache size
    fp16_memory_bytes: int         # FP16 reference cache size
    memory_savings_ratio: float    # fp16 / compressed

    # Accuracy
    max_abs_error: float           # max |score_quant - score_fp16|
    mean_abs_error: float          # mean |score_quant - score_fp16|
    cosine_similarity: float       # cos(output_quant, output_fp16)

    # Composite
    composite_score: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def reference_attention(queries, keys, values):
    """FP16 reference attention for accuracy comparison."""
    scale = queries.shape[-1] ** -0.5
    scores = (queries @ mx.transpose(keys, axes=(0, 1, 3, 2))) * scale
    weights = mx.softmax(scores, axis=-1)
    output = weights @ values
    return scores, output


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    return (dot / (norm_a * norm_b + 1e-8)).item()


def benchmark(
    embed_dim: int = 128,
    num_heads: int = 8,
    seq_len: int = 4096,
    batch_size: int = 1,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""

    config = TurboQuantConfig(embed_dim=embed_dim, num_heads=num_heads)

    B, H, N, D = batch_size, num_heads, seq_len, embed_dim
    keys = mx.random.normal((B, H, N, D))
    values = mx.random.normal((B, H, N, D))
    queries = mx.random.normal((B, H, 1, D))

    # Reference attention
    ref_scores, ref_output = reference_attention(queries, keys, values)
    mx.eval(ref_scores, ref_output)

    # TurboQuant attention
    cache = TurboQuantKVCache(config)
    cache.update(keys, values)

    # Warmup
    for _ in range(n_warmup):
        out = cache.attention(queries)
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = cache.attention(queries)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    quant_scores = cache.qjl.score(queries, cache.keys)
    mx.eval(quant_scores)

    # Accuracy
    score_diff = mx.abs(quant_scores - ref_scores)
    max_err = mx.max(score_diff).item()
    mean_err = mx.mean(score_diff).item()
    cos_sim = cosine_sim(out, ref_output)

    # Performance
    avg_time = np.median(times)
    tps = seq_len / avg_time
    mem = cache.memory_bytes()
    fp16_mem = cache.fp16_equivalent_bytes()
    savings = fp16_mem / max(mem, 1)

    # Composite score
    tolerance = 0.01
    error_penalty = 1.0 - min(max_err / tolerance, 1.0)
    composite = tps * savings * error_penalty

    return BenchmarkResult(
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_len=seq_len,
        batch_size=batch_size,
        throughput_tps=tps,
        kernel_time_ms=avg_time * 1000,
        memory_bytes=mem,
        fp16_memory_bytes=fp16_mem,
        memory_savings_ratio=savings,
        max_abs_error=max_err,
        mean_abs_error=mean_err,
        cosine_similarity=cos_sim,
        composite_score=composite,
    )


if __name__ == "__main__":
    print("TurboQuant Benchmark")
    print("=" * 60)

    configs = [
        {"seq_len": 1024, "embed_dim": 128, "num_heads": 8},
        {"seq_len": 4096, "embed_dim": 128, "num_heads": 8},
        {"seq_len": 16384, "embed_dim": 128, "num_heads": 8},
        {"seq_len": 65536, "embed_dim": 128, "num_heads": 8},
    ]

    results = []
    for cfg in configs:
        print(f"\nseq_len={cfg['seq_len']}, embed_dim={cfg['embed_dim']}, heads={cfg['num_heads']}")
        result = benchmark(**cfg)
        results.append(result)
        print(f"  Throughput:    {result.throughput_tps:,.0f} tokens/sec")
        print(f"  Kernel time:   {result.kernel_time_ms:.2f} ms")
        print(f"  Memory saving: {result.memory_savings_ratio:.1f}x")
        print(f"  Max error:     {result.max_abs_error:.6f}")
        print(f"  Cosine sim:    {result.cosine_similarity:.6f}")
        print(f"  Composite:     {result.composite_score:,.0f}")

    # Write results
    with open("benchmarks/latest.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults written to benchmarks/latest.json")
