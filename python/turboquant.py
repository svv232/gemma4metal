"""TurboQuant: Fused int4 SDPA for MLX.

Custom Metal kernel that computes attention directly from int4 quantized K/V
without materializing the dequantized matrices. 35% faster than dequantize+SDPA
at 786 tokens on M1 Max.

Usage:
    from turboquant import fused_int4_sdpa
    # Replace mlx-lm's attention with fused version
    attn_out = fused_int4_sdpa(queries, k_quant, k_scales, k_biases,
                                v_quant, v_scales, v_biases, gqa_factor)
"""
import mlx.core as mx

# Fused int4 SDPA kernel source (vectorized, MLX qdot pattern)
_SDPA_INT4_HEADER = """
// Dequantize int4 from uint16: 4 nibbles per uint16
// Pre-scaled query avoids per-nibble bit shifting
"""

_SDPA_INT4_SOURCE = """
    uint head_idx = thread_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    const int N = params[0];
    const int gqa = params[1];
    const int sliding_window_val = params[2];
    const int D = params[3];
    const float scale = scale_val[0];
    const int BD = 32;
    const int elems_per_lane = D / BD;
    const int BN = threads_per_threadgroup.x / BD;
    const int kv_head_idx = head_idx / gqa;

    const int k_packed_dim = D / 8;
    const int k_scale_dim = D / 64;
    int lane_start = simd_lid * elems_per_lane;

    // Pre-scale query for vectorized dot product
    float q_scaled[16];  // max elems_per_lane for D<=512
    for (int i = 0; i < elems_per_lane; i += 4) {
        float q0 = scale * queries[head_idx * D + lane_start + i];
        float q1 = scale * queries[head_idx * D + lane_start + i + 1];
        float q2 = scale * queries[head_idx * D + lane_start + i + 2];
        float q3 = scale * queries[head_idx * D + lane_start + i + 3];
        q_scaled[i]     = q0;
        q_scaled[i + 1] = q1 / 16.0f;
        q_scaled[i + 2] = q2 / 256.0f;
        q_scaled[i + 3] = q3 / 4096.0f;
    }

    float o[16];
    for (int i = 0; i < elems_per_lane; i++) o[i] = 0.0f;
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    const device uint32_t* kq_base = k_quant + kv_head_idx * N * k_packed_dim;
    const device float* ks_base = k_scales + kv_head_idx * N * k_scale_dim;
    const device float* kb_base = k_biases + kv_head_idx * N * k_scale_dim;
    const device uint32_t* vq_base = v_quant + kv_head_idx * N * k_packed_dim;
    const device float* vs_base = v_scales + kv_head_idx * N * k_scale_dim;
    const device float* vb_base = v_biases + kv_head_idx * N * k_scale_dim;

    for (int ki = simd_gid; ki < N; ki += BN) {
        if (sliding_window_val > 0 && (N - 1 - ki) >= sliding_window_val) continue;

        // Vectorized K scoring
        float score = 0.0f;
        const device uint16_t* kw = (const device uint16_t*)(kq_base + ki * k_packed_dim) + lane_start / 4;
        for (int i = 0; i < elems_per_lane; i += 4) {
            int group_idx = (lane_start + i) / 64;
            float s = ks_base[ki * k_scale_dim + group_idx];
            float b = kb_base[ki * k_scale_dim + group_idx];
            uint16_t w = kw[i / 4];
            float accum = q_scaled[i] * float(w & 0x000Fu)
                        + q_scaled[i+1] * float(w & 0x00F0u)
                        + q_scaled[i+2] * float(w & 0x0F00u)
                        + q_scaled[i+3] * float(w & 0xF000u);
            float qsum = q_scaled[i] + q_scaled[i+1]*16.0f + q_scaled[i+2]*256.0f + q_scaled[i+3]*4096.0f;
            score += s * accum + b * qsum;
        }
        score = simd_sum(score);

        float new_max = max(max_score, score);
        float factor = metal::fast::exp(max_score - new_max);
        float exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        // Vectorized V dequantize + accumulate
        const device uint16_t* vw = (const device uint16_t*)(vq_base + ki * k_packed_dim) + lane_start / 4;
        for (int i = 0; i < elems_per_lane; i += 4) {
            int group_idx = (lane_start + i) / 64;
            float vs = vs_base[ki * k_scale_dim + group_idx];
            float vb = vb_base[ki * k_scale_dim + group_idx];
            uint16_t w = vw[i / 4];
            float v0 = vs * float(w & 0x000Fu) + vb;
            float v1 = vs * float((w >> 4) & 0xFu) + vb;
            float v2 = vs * float((w >> 8) & 0xFu) + vb;
            float v3 = vs * float((w >> 12) & 0xFu) + vb;
            o[i]     = o[i]     * factor + exp_score * v0;
            o[i+1]   = o[i+1]  * factor + exp_score * v1;
            o[i+2]   = o[i+2]  * factor + exp_score * v2;
            o[i+3]   = o[i+3]  * factor + exp_score * v3;
        }
    }

    // Reduction
    threadgroup float tg_max_arr[32];
    threadgroup float tg_sum_arr[32];
    threadgroup float tg_accum[512];  // max D

    if (simd_lid == 0) {
        tg_max_arr[simd_gid] = max_score;
        tg_sum_arr[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sg_max = (simd_lid < uint(BN)) ? tg_max_arr[simd_lid] : -1e30f;
    float new_max_g = simd_max(sg_max);
    float my_factor = metal::fast::exp(max_score - new_max_g);
    float sg_sum_val = (simd_lid < uint(BN)) ? tg_sum_arr[simd_lid] : 0.0f;
    float sg_factor = metal::fast::exp(sg_max - new_max_g);
    float global_sum = simd_sum(sg_sum_val * sg_factor);

    // Sequential reduction via threadgroup accumulator
    for (int i = 0; i < elems_per_lane; i++) {
        o[i] *= my_factor;
    }
    if (simd_gid == 0) {
        for (int i = 0; i < elems_per_lane; i++)
            tg_accum[simd_lid * elems_per_lane + i] = o[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int sg = 1; sg < BN; sg++) {
        if (int(simd_gid) == sg) {
            for (int i = 0; i < elems_per_lane; i++)
                tg_accum[simd_lid * elems_per_lane + i] += o[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_gid == 0) {
        for (int i = 0; i < elems_per_lane; i++) {
            int elem = simd_lid * elems_per_lane + i;
            out[head_idx * D + elem] = global_sum == 0 ? 0.0f : (tg_accum[elem] / global_sum);
        }
    }
"""


def fused_int4_sdpa(
    queries,      # (num_heads, head_dim) float32
    k_quant,      # (num_kv_heads, N, head_dim/8) uint32
    k_scales,     # (num_kv_heads, N, head_dim/64) float32
    k_biases,     # (num_kv_heads, N, head_dim/64) float32
    v_quant,      # (num_kv_heads, N, head_dim/8) uint32
    v_scales,     # (num_kv_heads, N, head_dim/64) float32
    v_biases,     # (num_kv_heads, N, head_dim/64) float32
    gqa_factor,   # num_heads / num_kv_heads
    scale=1.0,    # attention scale (1.0 for Gemma 4)
    sliding_window=0,
):
    """Fused int4 SDPA: scores from compressed K, online softmax, V accumulation.

    35% faster than mx.dequantize + mx.fast.scaled_dot_product_attention
    at 786 tokens on M1 Max.
    """
    import struct

    num_heads = queries.shape[0]
    head_dim = queries.shape[1]
    N = k_quant.shape[1]

    # Pack params as int32 array (N, gqa, sliding_window, head_dim)
    params = mx.array([N, gqa_factor, sliding_window, head_dim], dtype=mx.int32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    # mx.fast.metal_kernel has 768 thread limit (vs 1024 for compiled metallib)
    BN = 24 if head_dim <= 256 else 16

    kernel = mx.fast.metal_kernel(
        name=f"sdpa_int4_fused_{head_dim}",
        input_names=["queries", "k_quant", "k_scales", "k_biases",
                      "v_quant", "v_scales", "v_biases", "params", "scale_val"],
        output_names=["out"],
        source=_SDPA_INT4_SOURCE,
        header=_SDPA_INT4_HEADER,
    )

    out = kernel(
        inputs=[queries, k_quant, k_scales, k_biases,
                v_quant, v_scales, v_biases, params, scale_arr],
        output_shapes=[(num_heads, head_dim)],
        output_dtypes=[mx.float32],
        grid=(num_heads, 1, 1),
        threadgroup=(BN * 32, 1, 1),
    )

    return out[0]
