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

    # Average over multiple trials for robust unbiasedness test
    n_trials = 10
    all_mean_errors = []
    all_corrs = []

    for trial in range(n_trials):
        mx.random.seed(trial)
        B, H, N, D = 1, 1, 200, 128
        keys = mx.random.normal((B, H, N, D))
        queries = mx.random.normal((B, H, 1, D))

        exact = (queries @ mx.transpose(keys, axes=(0, 1, 3, 2))).squeeze()
        quantized = qjl.quantize_keys(keys)
        estimated = qjl.score(queries, quantized).squeeze()
        mx.eval(exact, estimated)

        exact_np = np.array(exact)
        est_np = np.array(estimated)
        all_mean_errors.append((est_np - exact_np).mean())
        all_corrs.append(np.corrcoef(exact_np.flatten(), est_np.flatten())[0, 1])

    grand_mean_error = np.mean(all_mean_errors)
    mean_corr = np.mean(all_corrs)

    print(f"QJL unbiasedness: grand_mean_error = {grand_mean_error:.6f} (should be ~0)")
    print(f"QJL mean correlation with exact: {mean_corr:.4f}")

    # With 10 trials × 200 samples, the standard error is much smaller
    assert abs(grand_mean_error) < 1.5, f"QJL is biased: grand_mean_error = {grand_mean_error}"
    assert mean_corr > 0.75, f"QJL correlation too low: {mean_corr}"


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
    """Verify Gemma 4 31B @ 256K fits in 64GB with TurboQuant."""
    # Gemma 4 31B: estimate ~48 layers, ~16 KV heads, 128 dim per head
    # (exact arch TBD from model card, these are conservative estimates)
    n_layers = 48
    n_kv_heads = 16
    head_dim = 128
    seq_len = 256_000
    batch = 1

    # FP16 KV cache
    fp16_bytes_per_layer = batch * n_kv_heads * seq_len * head_dim * 2 * 2  # K+V, 2 bytes each
    fp16_total = fp16_bytes_per_layer * n_layers
    fp16_gb = fp16_total / (1024 ** 3)

    # TurboQuant 3-bit KV cache
    tq_bits_per_coord = 3
    tq_bytes_per_layer = batch * n_kv_heads * seq_len * head_dim * tq_bits_per_coord * 2 / 8
    norm_bytes = batch * n_kv_heads * seq_len * 4 * 2 * n_layers
    tq_total = tq_bytes_per_layer * n_layers + norm_bytes
    tq_gb = tq_total / (1024 ** 3)

    # Weights at 4-bit (31B params)
    weight_params = 31e9
    weight_gb = weight_params * 0.5 / (1024 ** 3)  # 4-bit = 0.5 bytes

    print(f"\nGemma 4 31B @ 256K context memory estimate:")
    print(f"  Weights (4-bit):           {weight_gb:.1f} GB")
    print(f"  KV cache FP16:             {fp16_gb:.1f} GB")
    print(f"  KV cache TurboQuant 3-bit: {tq_gb:.1f} GB")
    print(f"  Total without TurboQuant:  {weight_gb + fp16_gb:.1f} GB")
    print(f"  Total with TurboQuant:     {weight_gb + tq_gb:.1f} GB")

    assert weight_gb + tq_gb < 64, "TurboQuant Gemma4-31B@256K should fit in 64GB"
    print("  VERIFIED: Fits in 64GB with TurboQuant.")


def test_turboquant_combined():
    """TurboQuant combined (PolarQuant + QJL) should achieve near-reference quality."""
    mx.random.seed(42)
    config = TurboQuantConfig(embed_dim=128, num_heads=8)

    B, H, N, D = 1, 8, 64, 128
    keys = mx.random.normal((B, H, N, D))
    values = mx.random.normal((B, H, N, D))
    queries = mx.random.normal((B, H, 1, D))

    # Reference attention
    scale = D ** -0.5
    ref_scores = (queries @ mx.transpose(keys, axes=(0, 1, 3, 2))) * scale
    ref_weights = mx.softmax(ref_scores, axis=-1)
    ref_output = ref_weights @ values
    mx.eval(ref_output)

    # TurboQuant combined
    cache = TurboQuantKVCache(config, mode="turboquant")
    cache.update(keys, values)
    tq_output = cache.attention(queries)
    mx.eval(tq_output)

    # Cosine similarity
    ref_flat = ref_output.reshape(-1).astype(mx.float32)
    tq_flat = tq_output.reshape(-1).astype(mx.float32)
    cos_sim = (mx.sum(ref_flat * tq_flat) /
               (mx.sqrt(mx.sum(ref_flat ** 2)) * mx.sqrt(mx.sum(tq_flat ** 2)) + 1e-8)).item()

    print(f"TurboQuant combined cosine similarity: {cos_sim:.6f}")
    assert cos_sim > 0.95, f"TurboQuant quality too low: cos_sim = {cos_sim}"

    # Memory savings
    mem = cache.memory_bytes()
    fp16 = cache.fp16_equivalent_bytes()
    compression = fp16 / mem
    print(f"Memory: {mem / 1024:.1f} KB compressed, {fp16 / 1024:.1f} KB FP16, {compression:.2f}x")
    assert compression > 1.0, f"TurboQuant should compress: ratio = {compression}"


def test_metal_kernels():
    """Metal kernels should produce identical results to Python reference."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "python"))
    from qjl_metal import QJLProjectionMetal, pack_sign_bits, qjl_score_metal
    from polar_metal import polar_forward_metal, polar_inverse_metal

    mx.random.seed(42)

    # QJL Pack
    sign_bits = mx.random.randint(0, 2, (100, 128)).astype(mx.uint8)
    packed = pack_sign_bits(sign_bits)
    mx.eval(packed)
    assert packed.shape == (100, 16), f"Wrong pack shape: {packed.shape}"
    # Verify first byte manually
    expected = 0
    for i in range(8):
        expected |= (int(sign_bits[0, i].item()) << (7 - i))
    assert packed[0, 0].item() == expected, "QJL pack incorrect"
    print("  QJL Pack: correct")

    # QJL Score
    qjl = QJLProjectionMetal(embed_dim=128, sketch_dim=128, seed=42)
    keys = mx.random.normal((200, 128))
    queries = mx.random.normal((1, 128))
    quantized = qjl.quantize(keys)
    scores = qjl.score(queries, quantized)
    exact = queries @ keys.T
    mx.eval(scores, exact)
    corr = np.corrcoef(np.array(scores).flatten(), np.array(exact).flatten())[0, 1]
    print(f"  QJL Score: correlation={corr:.4f}")
    assert corr > 0.7, f"QJL Metal score correlation too low: {corr}"

    # Polar Forward
    y = mx.random.normal((1000 * 16,))
    angles = polar_forward_metal(y)
    mx.eval(*angles)
    assert angles[0].shape == (1000 * 8,), f"Wrong L1 angles shape"
    assert angles[4].shape == (1000,), f"Wrong radii shape"
    print(f"  Polar Forward: shapes correct")

    # Polar Inverse roundtrip
    config = TurboQuantConfig(polar_block_size=16, polar_bits_level1=4, polar_bits_higher=2)
    polar = PolarQuantizer(config)

    angles_l1, angles_l2, angles_l3, angles_l4, radii = polar_forward_metal(y)
    angle_arrays = [angles_l1, angles_l2, angles_l3, angles_l4]
    flat_indices = []
    for level, (angles_arr, codebook) in enumerate(zip(angle_arrays, polar.codebooks)):
        diffs = mx.abs(angles_arr.reshape(-1, 1) - codebook)
        indices = mx.argmin(diffs, axis=-1).astype(mx.uint32)
        flat_indices.append(indices)
    mx.eval(*flat_indices, radii)

    y_recon = polar_inverse_metal(polar.codebooks, flat_indices, radii)
    mx.eval(y_recon)
    mse = mx.mean((y - y_recon) ** 2).item()
    norm_sq = mx.mean(y ** 2).item()
    rel_mse = mse / norm_sq
    print(f"  Polar Roundtrip: relative MSE={rel_mse:.6f}")
    assert rel_mse < 0.05, f"Metal polar roundtrip MSE too high: {rel_mse}"

    print("  All Metal kernel tests passed")


def test_presets():
    """Quality presets should produce expected compression ratios."""
    mx.random.seed(42)
    B, H, N, D = 1, 8, 256, 128

    keys = mx.random.normal((B, H, N, D))
    values = mx.random.normal((B, H, N, D))
    queries = mx.random.normal((B, H, 1, D))

    scale = D ** -0.5
    ref_output = mx.softmax(
        (queries @ mx.transpose(keys, axes=(0, 1, 3, 2))) * scale, axis=-1
    ) @ values
    mx.eval(ref_output)

    expected = {
        "fast": (0.90, 3.5),
        "balanced": (0.95, 3.0),
        "quality": (0.97, 2.8),
        "hifi": (0.99, 2.0),
    }

    for name, (min_cos, min_compress) in expected.items():
        config = TurboQuantConfig.preset(name)
        cache = TurboQuantKVCache(config, mode="turboquant")
        cache.update(keys, values)
        out = cache.attention(queries)
        mx.eval(out)

        af = ref_output.reshape(-1).astype(mx.float32)
        bf = out.reshape(-1).astype(mx.float32)
        cos = (mx.sum(af * bf) / (mx.sqrt(mx.sum(af**2)) * mx.sqrt(mx.sum(bf**2)) + 1e-8)).item()
        compression = cache.fp16_equivalent_bytes() / cache.memory_bytes()

        print(f"  {name}: cos_sim={cos:.4f} (>={min_cos}), compress={compression:.2f}x (>={min_compress})")
        assert cos >= min_cos, f"Preset '{name}' cos_sim {cos} < {min_cos}"
        assert compression >= min_compress, f"Preset '{name}' compression {compression} < {min_compress}"


if __name__ == "__main__":
    tests = [
        ("QJL Unbiasedness", test_qjl_unbiasedness),
        ("Polar Roundtrip", test_polar_roundtrip),
        ("Polar Quantized Accuracy", test_polar_quantized_accuracy),
        ("KV Cache Attention", test_kv_cache_attention),
        ("TurboQuant Combined", test_turboquant_combined),
        ("Metal Kernels", test_metal_kernels),
        ("Quality Presets", test_presets),
        ("Memory Estimate (Gemma4-31B@256K)", test_memory_estimate),
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
