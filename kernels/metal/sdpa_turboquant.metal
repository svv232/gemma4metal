// TurboQuant Compressed-Domain SDPA Vector Kernel
//
// Forked from MLX's sdpa_vector.h with ONE surgical change:
// Instead of reading raw key floats, we reconstruct keys on-the-fly
// from PolarQuant compressed indices + radii.
//
// The rest (online softmax, value accumulation, SIMD reduction) is IDENTICAL
// to MLX's production flash attention.
//
// Memory bandwidth savings: reads ~4 bytes per key coordinate instead of 2 (FP16)
// with uint8 indices, or ~0.5 bytes with bit-packed indices.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Helper: reconstruct one block of 16 coordinates from polar indices
inline void polar_inverse_block(
    thread float* r,             // output: 16 floats
    const device float* cb_l1,
    const device float* cb_l2,
    const device float* cb_l3,
    const device float* cb_l4,
    const device uint8_t* idx_l1, // 8 indices for this block
    const device uint8_t* idx_l2, // 4 indices
    const device uint8_t* idx_l3, // 2 indices
    uint8_t idx_l4,               // 1 index
    float radius                  // block radius
) {
    r[0] = radius;

    // Level 4: 1 → 2
    {
        float theta = cb_l4[idx_l4];
        float r0 = r[0];
        r[0] = r0 * metal::cos(theta);
        r[1] = r0 * metal::sin(theta);
    }
    // Level 3: 2 → 4
    {
        float tmp[4];
        for (uint j = 0; j < 2; j++) {
            float theta = cb_l3[idx_l3[j]];
            tmp[2*j]   = r[j] * metal::cos(theta);
            tmp[2*j+1] = r[j] * metal::sin(theta);
        }
        for (uint j = 0; j < 4; j++) r[j] = tmp[j];
    }
    // Level 2: 4 → 8
    {
        float tmp[8];
        for (uint j = 0; j < 4; j++) {
            float theta = cb_l2[idx_l2[j]];
            tmp[2*j]   = r[j] * metal::cos(theta);
            tmp[2*j+1] = r[j] * metal::sin(theta);
        }
        for (uint j = 0; j < 8; j++) r[j] = tmp[j];
    }
    // Level 1: 8 → 16
    {
        float tmp[16];
        for (uint j = 0; j < 8; j++) {
            float theta = cb_l1[idx_l1[j]];
            tmp[2*j]   = r[j] * metal::cos(theta);
            tmp[2*j+1] = r[j] * metal::sin(theta);
        }
        for (uint j = 0; j < 16; j++) r[j] = tmp[j];
    }
}


// ──────────────────────────────────────────────────────────────────────
// Main kernel: sdpa_vector with compressed key loading
//
// Architecture matches MLX sdpa_vector exactly:
// - BN=32 simdgroups, each handles one key token
// - BD=32 SIMD lanes, each handles D/32 elements
// - Online softmax with running max + sum_exp
// - Threadgroup reduction for final output
//
// Changes from MLX sdpa_vector:
// - keys buffer replaced with compressed indices + radii + codebooks
// - Key loading (lines 109-111 in original) replaced with polar_inverse
// - Values still read as raw floats (or could be mx.dequantize'd)
// ──────────────────────────────────────────────────────────────────────

struct TQParams {
    int N;            // number of key tokens
    int D;            // head dimension
    int n_blocks;     // D / 16 (blocks per key vector)
    float scale;      // 1/sqrt(D)
    int gqa_factor;
};

kernel void sdpa_turboquant(
    // Query (full precision, preconditioned: Pq)
    const device float* queries        [[buffer(0)]],   // (B*H, Nq, D)
    // Compressed keys
    const device uint8_t* indices_l1   [[buffer(1)]],   // (B_kv*H_kv, N, n_blocks*8)
    const device uint8_t* indices_l2   [[buffer(2)]],   // (B_kv*H_kv, N, n_blocks*4)
    const device uint8_t* indices_l3   [[buffer(3)]],   // (B_kv*H_kv, N, n_blocks*2)
    const device uint8_t* indices_l4   [[buffer(4)]],   // (B_kv*H_kv, N, n_blocks)
    const device half* radii           [[buffer(5)]],   // (B_kv*H_kv, N, n_blocks)
    // Codebooks (shared across all tokens)
    const device float* cb_l1          [[buffer(6)]],
    const device float* cb_l2          [[buffer(7)]],
    const device float* cb_l3          [[buffer(8)]],
    const device float* cb_l4          [[buffer(9)]],
    // Values (full precision or mx.dequantize'd)
    const device float* values         [[buffer(10)]],  // (B_kv*H_kv, N, D)
    // Output
    device float* out                  [[buffer(11)]],  // (B*H, Nq, D)
    // Parameters
    constant TQParams& params          [[buffer(12)]],
    // Thread indexing
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr int BN = 32;  // key tokens per threadgroup (= number of simdgroups)
    constexpr int BD = 32;  // SIMD width

    const int D = params.D;
    const int N = params.N;
    const int n_blocks = params.n_blocks;
    const float scale = params.scale;
    const int gqa_factor = params.gqa_factor;

    const int qk_per_thread = D / BD;  // elements per SIMD lane
    const int v_per_thread = D / BD;

    // Thread-local registers
    float q[8];   // max D/32 = 256/32 = 8
    float k_recon[8];
    float o[8];

    // Threadgroup shared memory for reduction
    threadgroup float outputs_tg[BN * BD];
    threadgroup float max_scores_tg[BN];
    threadgroup float sum_exp_scores_tg[BN];

    // Position setup
    const int q_batch_head_idx = tid.x;
    const int q_seq_idx = tid.y;
    const int kv_head_idx = q_batch_head_idx / gqa_factor;

    // Query pointer: this thread reads qk_per_thread elements starting at simd_lid * qk_per_thread
    const int q_offset = (q_batch_head_idx * tpg.y + q_seq_idx) * D + simd_lid * qk_per_thread;

    // Key indices: per KV head
    const int kv_base = kv_head_idx * N;  // base offset for this KV head

    // Value pointer
    const int v_base = kv_head_idx * N * D;

    // Output pointer
    const int o_offset = (q_batch_head_idx * tpg.y + q_seq_idx) * D;

    // Read query (pre-scaled)
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = scale * queries[q_offset + i];
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0.0f;
    }

    float max_score = -1e30f;
    float sum_exp_score = 0.0f;

    // ── Main loop: for each key token ──
    // Each simdgroup handles one key token, striding by BN
    for (int ki = simd_gid; ki < N; ki += BN) {
        // ── TURBOQUANT: reconstruct key from compressed indices ──
        // Each SIMD lane needs its portion of the key: elements [simd_lid * qk_per_thread, ...)
        // These elements span across polar blocks.

        // Which blocks does this lane's elements fall in?
        int elem_start = simd_lid * qk_per_thread;

        // Reconstruct the full key into a shared buffer? No — each lane only needs its portion.
        // For block_size=16: block b covers elements [b*16, (b+1)*16)
        // Lane's elements [elem_start, elem_start + qk_per_thread) may span 1-2 blocks.

        // Simple approach: reconstruct each block that overlaps with this lane's elements
        for (int i = 0; i < qk_per_thread; i++) {
            int elem = elem_start + i;
            int blk = elem / 16;
            int blk_offset = elem % 16;

            // Only reconstruct if we're at the start of a block (avoid redundant work)
            if (blk_offset == 0 || i == 0) {
                int block_idx = (kv_base + ki) * n_blocks + blk;
                float r[16];
                polar_inverse_block(
                    r, cb_l1, cb_l2, cb_l3, cb_l4,
                    indices_l1 + block_idx * 8,
                    indices_l2 + block_idx * 4,
                    indices_l3 + block_idx * 2,
                    indices_l4[block_idx],
                    static_cast<float>(radii[block_idx])
                );
                // Cache reconstructed block in thread-local
                // (only the elements we need)
                for (int j = 0; j < qk_per_thread && (blk_offset + j) < 16; j++) {
                    if (elem_start + j >= blk * 16 && elem_start + j < (blk + 1) * 16) {
                        k_recon[j] = r[elem_start + j - blk * 16];
                    }
                }
            }
        }

        // ── Score: q · k (same as original) ──
        float score = 0.0f;
        for (int i = 0; i < qk_per_thread; i++) {
            score += q[i] * k_recon[i];
        }
        score = simd_sum(score);

        // ── Online softmax (identical to MLX) ──
        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // ── Value accumulation (identical to MLX) ──
        int v_offset = v_base + ki * D + simd_lid * v_per_thread;
        for (int i = 0; i < v_per_thread; i++) {
            o[i] = o[i] * factor + exp_score * values[v_offset + i];
        }
    }

    // ── Reduction (identical to MLX sdpa_vector) ──
    if (simd_lid == 0) {
        max_scores_tg[simd_gid] = max_score;
        sum_exp_scores_tg[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores_tg[simd_lid];
    float new_max = simd_max(max_score);
    float factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores_tg[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs_tg[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs_tg[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write output ──
    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            out[o_offset + simd_gid * v_per_thread + i] = o[i];
        }
    }
}
