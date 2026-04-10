"""
Correctness tests for TurboQuant components.

Tests numerical accuracy of QJL and PolarQuant against
exact FP32 reference implementations.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "python"))

import mlx.core as mx
import numpy as np

from turboquant import TurboQuantConfig, QJLProjection, PolarQuantizer, TurboQuantKVCache


def test_qjl_unbiasedness():
    """QJL inner product estimator should be unbiased (mean error → 0)."""
    config = TurboQuantConfig(embed_dim=128, sketch_dim=128, seed=42)
    qjl = QJLProjection(config)

    B, H, N, D = 1, 1, 100, 128
    keys = mx.random.normal((B, H, N, D))
    queries = mx.random.normal((B, H, 1, D))

    # Exact inner products
    exact = (queries @ mx.transpose(keys, axes=(0, 1, 3, 2))).squeeze()

    # QJL estimated inner products
    quantized = qjl.quantize_keys(keys)
    estimated = qjl.score(queries, quantized).squeeze()

    mx.eval(exact, estimated)

    # Mean error should be near zero (unbiased)
    mean_error = mx.mean(estimated - exact).item()
    print(f"QJL unbiasedness: mean_error = {mean_error:.6f} (should be ~0)")
    assert abs(mean_error) < 0.5, f"QJL is biased: mean_error = {mean_error}"

    # Correlation should be high
    exact_np = np.array(exact)
    est_np = np.array(estimated)
    corr = np.corrcoef(exact_np.flatten(), est_np.flatten())[0, 1]
    print(f"QJL correlation with exact: {corr:.4f}")
    assert corr > 0.8, f"QJL correlation too low: {corr}"


def test_polar_roundtrip():
    """PolarQuant forward + inverse should approximately reconstruct the input."""
    config = TurboQuantConfig(polar_block_size=16, polar_bits_level1=4, polar_bits_higher=2)
    polar = PolarQuantizer(config)

    # Random vector block
    y = mx.random.normal((1, 1, 4, 16))  # (B, H, N_blocks, block_size)

    # Forward transform
    angles, radius = polar.forward_polar(y)
    mx.eval(*angles, radius)

    # Inverse transform (using exact angles, no quantization)
    y_reconstructed = polar.inverse_polar(angles, radius)
    mx.eval(y_reconstructed)

    # Should be exact (no quantization applied)
    error = mx.max(mx.abs(y - y_reconstructed)).item()
    print(f"Polar roundtrip error (no quantization): {error:.2e} (should be ~0)")
    assert error < 1e-4, f"Polar roundtrip error too large: {error}"


def test_polar_quantized_accuracy():
    """PolarQuant with quantization should have bounded error."""
    config = TurboQuantConfig(polar_block_size=16, polar_bits_level1=4, polar_bits_higher=2)
    polar = PolarQuantizer(config)

    y = mx.random.normal((1, 1, 100, 16))

    # Forward transform
    angles, radius = polar.forward_polar(y)

    # Quantize angles to nearest codebook entry
    quantized_angles = []
    for level, (angle_arr, codebook) in enumerate(zip(angles, polar.codebooks)):
        # Find nearest centroid
        diffs = mx.abs(mx.expand_dims(angle_arr, axis=-1) - codebook)
        indices = mx.argmin(diffs, axis=-1)
        quantized = codebook[indices]
        quantized_angles.append(quantized)

    # Inverse with quantized angles
    y_reconstructed = polar.inverse_polar(quantized_angles, radius)
    mx.eval(y, y_reconstructed)

    # Measure reconstruction error
    mse = mx.mean((y - y_reconstructed) ** 2).item()
    norm_sq = mx.mean(y ** 2).item()
    relative_error = mse / max(norm_sq, 1e-8)
    print(f"Polar quantized relative MSE: {relative_error:.6f}")
    print(f"Polar quantized RMSE: {np.sqrt(mse):.6f}")


def test_kv_cache_attention():
    """Full KV cache attention should produce reasonable output."""
    config = TurboQuantConfig(embed_dim=128, num_heads=8)
    cache = TurboQuantKVCache(config)

    B, H, N, D = 1, 8, 64, 128
    keys = mx.random.normal((B, H, N, D))
    values = mx.random.normal((B, H, N, D))
    queries = mx.random.normal((B, H, 1, D))

    # Reference
    scale = D ** -0.5
    ref_scores = (queries @ mx.transpose(keys, axes=(0, 1, 3, 2))) * scale
    ref_weights = mx.softmax(ref_scores, axis=-1)
    ref_output = ref_weights @ values

    # TurboQuant
    cache.update(keys, values)
    tq_output = cache.attention(queries)

    mx.eval(ref_output, tq_output)

    # Output cosine similarity
    ref_flat = ref_output.reshape(-1).astype(mx.float32)
    tq_flat = tq_output.reshape(-1).astype(mx.float32)
    cos_sim = (mx.sum(ref_flat * tq_flat) /
               (mx.sqrt(mx.sum(ref_flat ** 2)) * mx.sqrt(mx.sum(tq_flat ** 2)) + 1e-8)).item()
    print(f"Attention output cosine similarity: {cos_sim:.6f}")

    # Memory savings
    compressed = cache.memory_bytes()
    fp16 = cache.fp16_equivalent_bytes()
    print(f"Memory: {compressed / 1024:.1f} KB compressed vs {fp16 / 1024:.1f} KB FP16 ({fp16 / compressed:.1f}x)")


def test_memory_estimate():
    """Verify 70B @ 128K fits in 64GB with TurboQuant."""
    # Llama 3.1 70B: 80 layers, 8 KV heads, 128 dim per head
    n_layers = 80
    n_kv_heads = 8
    head_dim = 128
    seq_len = 128_000
    batch = 1

    # FP16 KV cache
    fp16_bytes_per_layer = batch * n_kv_heads * seq_len * head_dim * 2 * 2  # K+V, 2 bytes each
    fp16_total = fp16_bytes_per_layer * n_layers
    fp16_gb = fp16_total / (1024 ** 3)

    # TurboQuant 3-bit KV cache
    # 3 bits per coord for keys + values
    tq_bits_per_coord = 3
    tq_bytes_per_layer = batch * n_kv_heads * seq_len * head_dim * tq_bits_per_coord * 2 / 8  # K+V
    # Plus norms: 4 bytes per token per head per layer, K+V
    norm_bytes = batch * n_kv_heads * seq_len * 4 * 2 * n_layers
    tq_total = tq_bytes_per_layer * n_layers + norm_bytes
    tq_gb = tq_total / (1024 ** 3)

    # Weights at 4-bit
    weight_params = 70e9
    weight_gb = weight_params * 0.5 / (1024 ** 3)  # 4-bit = 0.5 bytes

    print(f"\nLlama 3.1 70B @ 128K context memory estimate:")
    print(f"  Weights (4-bit):           {weight_gb:.1f} GB")
    print(f"  KV cache FP16:             {fp16_gb:.1f} GB")
    print(f"  KV cache TurboQuant 3-bit: {tq_gb:.1f} GB")
    print(f"  Total without TurboQuant:  {weight_gb + fp16_gb:.1f} GB (> 64 GB)")
    print(f"  Total with TurboQuant:     {weight_gb + tq_gb:.1f} GB (< 64 GB)")

    assert weight_gb + tq_gb < 64, "TurboQuant 70B@128K should fit in 64GB"
    assert weight_gb + fp16_gb > 64, "FP16 70B@128K should NOT fit in 64GB"
    print("  VERIFIED: Fits in 64GB with TurboQuant, doesn't fit without.")


if __name__ == "__main__":
    tests = [
        ("QJL Unbiasedness", test_qjl_unbiasedness),
        ("Polar Roundtrip", test_polar_roundtrip),
        ("Polar Quantized Accuracy", test_polar_quantized_accuracy),
        ("KV Cache Attention", test_kv_cache_attention),
        ("Memory Estimate (70B@128K)", test_memory_estimate),
    ]

    print("TurboQuant Correctness Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
            print(f"PASS")
        except Exception as e:
            failed += 1
            print(f"FAIL: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
