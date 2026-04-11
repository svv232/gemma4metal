"""
Metal-accelerated QJL kernels via MLX custom Metal kernel API.

Two kernels:
1. qjl_pack: Pack sign bits into uint8 (after MLX-native matmul + sign)
2. qjl_score_metal: Fused unpack + inner product for scoring
"""

import mlx.core as mx
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Kernel 1: Pack sign bits into uint8
# Input: sign_bits (N, S) uint8 with values 0/1
# Output: packed (N, S/8) uint8 with 8 bits packed per byte
# ──────────────────────────────────────────────────────────────────────

_pack_kernel = mx.fast.metal_kernel(
    name="qjl_pack_bits",
    input_names=["sign_bits"],
    output_names=["packed"],
    source="""
        // Each thread packs 8 consecutive bits into 1 byte
        uint n = thread_position_in_grid.x;  // token index
        uint byte_idx = thread_position_in_grid.y;  // output byte index

        uint S = sign_bits_shape[1];  // sketch_dim
        uint base = n * S + byte_idx * 8;

        uint8_t result = 0;
        for (uint b = 0; b < 8; b++) {
            if (byte_idx * 8 + b < S) {
                result |= (sign_bits[base + b] & 1) << (7 - b);
            }
        }
        packed[n * (S / 8) + byte_idx] = result;
    """,
)


def pack_sign_bits(sign_bits: mx.array) -> mx.array:
    """Pack (N, S) uint8 sign bits into (N, S//8) packed uint8."""
    N, S = sign_bits.shape
    assert S % 8 == 0, f"sketch_dim must be divisible by 8, got {S}"
    outputs = _pack_kernel(
        inputs=[sign_bits],
        output_shapes=[(N, S // 8)],
        output_dtypes=[mx.uint8],
        grid=(N, S // 8, 1),
        threadgroup=(1, min(S // 8, 256), 1),
    )
    return outputs[0]


# ──────────────────────────────────────────────────────────────────────
# Kernel 2: QJL Score — fused unpack + inner product
# For each (query, key) pair:
#   score = sqrt(pi/2) / S * norm_k * Σ_i (bit_i ? +sketch_i : -sketch_i)
#
# Input: query_sketch (Nq, S), packed_bits (Nk, S/8), key_norms (Nk,)
# Output: scores (Nq, Nk)
#
# Grid: (Nq, Nk, 1), each thread computes one score
# ──────────────────────────────────────────────────────────────────────

_score_header = """
// sqrt(pi/2) ≈ 1.2533141
constant constexpr float QJL_SCALE_FACTOR = 1.2533141f;
"""

_score_kernel = mx.fast.metal_kernel(
    name="qjl_score",
    input_names=["query_sketch", "packed_bits", "key_norms"],
    output_names=["scores"],
    header=_score_header,
    source="""
        uint qi = thread_position_in_grid.x;  // query index
        uint ki = thread_position_in_grid.y;  // key index

        uint S = query_sketch_shape[1];       // sketch_dim
        uint bytes_per_key = S / 8;

        float inner = 0.0f;
        for (uint byte_idx = 0; byte_idx < bytes_per_key; byte_idx++) {
            uint8_t byte_val = packed_bits[ki * bytes_per_key + byte_idx];
            uint base_bit = byte_idx * 8;

            // Unroll 8 bits per byte
            for (uint b = 0; b < 8; b++) {
                uint bit_pos = base_bit + b;
                if (bit_pos < S) {
                    float sv = query_sketch[qi * S + bit_pos];
                    uint8_t bit = (byte_val >> (7 - b)) & 1;
                    inner += bit ? sv : -sv;
                }
            }
        }

        float scale = QJL_SCALE_FACTOR / float(S);
        float norm = key_norms[ki];
        scores[qi * key_norms_shape[0] + ki] = scale * norm * inner;
    """,
)


# ──────────────────────────────────────────────────────────────────────
# Kernel 2b: Tiled QJL Score with threadgroup memory for better perf
# Uses SIMD reductions and shared memory for the query sketch
# ──────────────────────────────────────────────────────────────────────

_score_tiled_kernel = mx.fast.metal_kernel(
    name="qjl_score_tiled",
    input_names=["query_sketch", "packed_bits", "key_norms"],
    output_names=["scores"],
    header=_score_header,
    source="""
        // Each group of 32 threads handles one (query, key) pair
        // Grid: (Nq * 32, Nk, 1), Threadgroup: (32, 1, 1)
        uint global_x = thread_position_in_grid.x;
        uint qi = global_x / 32;
        uint lane = global_x % 32;
        uint ki = thread_position_in_grid.y;

        uint S = query_sketch_shape[1];
        uint Nk = key_norms_shape[0];
        uint bytes_per_key = S / 8;

        if (ki >= Nk) return;

        // Each lane handles a portion of the bytes
        float inner = 0.0f;
        for (uint byte_idx = lane; byte_idx < bytes_per_key; byte_idx += 32) {
            uint8_t byte_val = packed_bits[ki * bytes_per_key + byte_idx];
            uint base_bit = byte_idx * 8;

            for (uint b = 0; b < 8; b++) {
                uint bit_pos = base_bit + b;
                if (bit_pos < S) {
                    float sv = query_sketch[qi * S + bit_pos];
                    uint8_t bit = (byte_val >> (7 - b)) & 1;
                    inner += bit ? sv : -sv;
                }
            }
        }

        // SIMD reduction across the 32 lanes
        inner = simd_sum(inner);

        if (lane == 0) {
            float scale = QJL_SCALE_FACTOR / float(S);
            float norm = key_norms[ki];
            scores[qi * Nk + ki] = scale * norm * inner;
        }
    """,
)


def qjl_score_metal(
    query_sketch: mx.array,
    packed_bits: mx.array,
    key_norms: mx.array,
    use_tiled: bool = True,
) -> mx.array:
    """
    Compute QJL attention scores using Metal kernels.

    Args:
        query_sketch: (Nq, S) float32 — q @ projection^T
        packed_bits: (Nk, S/8) uint8 — packed sign bits
        key_norms: (Nk,) float32 — key vector norms

    Returns:
        scores: (Nq, Nk) float32 — attention logits (unscaled by 1/sqrt(d))
    """
    Nq = query_sketch.shape[0]
    Nk = packed_bits.shape[0]

    if use_tiled:
        outputs = _score_tiled_kernel(
            inputs=[query_sketch, packed_bits, key_norms],
            output_shapes=[(Nq, Nk)],
            output_dtypes=[mx.float32],
            grid=(Nq * 32, Nk, 1),
            threadgroup=(32, 1, 1),
        )
    else:
        outputs = _score_kernel(
            inputs=[query_sketch, packed_bits, key_norms],
            output_shapes=[(Nq, Nk)],
            output_dtypes=[mx.float32],
            grid=(Nq, Nk, 1),
            threadgroup=(1, 1, 1),
        )
    return outputs[0]


class QJLProjectionMetal:
    """Metal-accelerated QJL projection and scoring."""

    def __init__(self, embed_dim: int, sketch_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        gaussian = rng.randn(sketch_dim, embed_dim).astype(np.float32)
        q, _ = np.linalg.qr(gaussian)
        self.projection = mx.array(q * np.sqrt(embed_dim))
        self.sketch_dim = sketch_dim
        self.embed_dim = embed_dim
        self.scale = np.sqrt(np.pi / 2) / sketch_dim

    def quantize(self, keys: mx.array) -> dict:
        """
        Quantize key vectors to packed sign bits.

        Args:
            keys: (..., embed_dim) — key vectors

        Returns:
            dict with packed_bits, norms, and original shape info
        """
        orig_shape = keys.shape
        # Flatten to (N, D) for kernel dispatch
        keys_flat = keys.reshape(-1, self.embed_dim)
        N = keys_flat.shape[0]

        # Project and take signs (MLX native — fast for matmul)
        sketch = keys_flat @ self.projection.T  # (N, sketch_dim)
        sign_bits = mx.where(
            sketch >= 0, mx.array(1, dtype=mx.uint8), mx.array(0, dtype=mx.uint8)
        )

        # Pack bits via Metal kernel
        packed = pack_sign_bits(sign_bits)

        # Compute norms
        norms = mx.sqrt(mx.sum(keys_flat * keys_flat, axis=-1))  # (N,)

        return {
            "packed_bits": packed,   # (N, sketch_dim/8)
            "norms": norms,          # (N,)
            "orig_shape": orig_shape,
            "N": N,
        }

    def score(self, queries: mx.array, quantized: dict) -> mx.array:
        """
        Compute QJL attention scores using Metal kernel.

        Args:
            queries: (..., embed_dim) — query vectors
            quantized: output of quantize()

        Returns:
            scores: (Nq, Nk) — raw inner product estimates
        """
        queries_flat = queries.reshape(-1, self.embed_dim)
        Nq = queries_flat.shape[0]

        # Compute query sketch (MLX native matmul)
        query_sketch = queries_flat @ self.projection.T  # (Nq, sketch_dim)

        # Score via Metal kernel
        scores = qjl_score_metal(
            query_sketch,
            quantized["packed_bits"],
            quantized["norms"],
        )

        return scores  # (Nq, Nk)
