// QJL Quantization & Scoring Kernels for Apple Silicon Metal
//
// Projects key vectors through random matrix S, takes sign bits, packs into uint8.
// Port of CUDA qjl_quant_kernel from github.com/amirzandieh/QJL
//
// Architecture: M1 Max, 32-core GPU, 32KB threadgroup memory, SIMD width 32

#include <metal_stdlib>
using namespace metal;

constant uint SIMD_WIDTH = 32;

struct QJLQuantParams {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint embed_dim;
    uint sketch_dim;
    uint num_outliers;
};

// ────────────────────���─────────────────────────────────────────────────
// qjl_quant: Project key vectors → sign bits → packed uint8
//
// Grid:  (B * H * N, ceil(sketch_dim / (SIMDGROUPS * 8)))
// Threadgroup: (SIMD_WIDTH, SIMDGROUPS) where SIMDGROUPS = 8
//
// Each threadgroup processes ONE key token for a block of projection rows.
// Each simdgroup handles one projection row → dot product → sign bit.
// 8 simdgroups produce 8 sign bits → 1 packed uint8.
// ──────────────────────────────────────────────────────────────────────

constant uint QUANT_SIMDGROUPS = 8;  // 8 bits per byte

kernel void qjl_quant(
    device const float* keys          [[buffer(0)]],   // (B, H, N, D)
    device const float* projection    [[buffer(1)]],   // (sketch_dim, embed_dim)
    device const uint8_t* outlier_mask [[buffer(2)]],  // (D,) — 1 if outlier channel
    device uint8_t* packed_bits       [[buffer(3)]],   // (B, H, N, sketch_dim/8)
    device float* norms               [[buffer(4)]],   // (B, H, N)
    constant QJLQuantParams& params   [[buffer(5)]],
    uint3 tg_pos    [[threadgroup_position_in_grid]],
    uint3 tid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // tg_pos.x = token index (flat: b * H * N + h * N + n)
    // tg_pos.y = output byte block index
    uint token_idx = tg_pos.x;
    uint byte_block = tg_pos.y;

    if (token_idx >= params.batch_size * params.num_heads * params.seq_len) return;

    uint D = params.embed_dim;
    uint bytes_per_token = params.sketch_dim / 8;

    // Base pointer for this token's key vector
    device const float* key_ptr = keys + token_idx * D;

    // Shared memory for the key vector tile
    threadgroup float shared_key[1024];  // max embed_dim = 1024

    // Step 1: Cooperatively load key vector into shared memory
    for (uint i = simd_group * SIMD_WIDTH + simd_lane; i < D;
         i += QUANT_SIMDGROUPS * SIMD_WIDTH) {
        shared_key[i] = key_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Each simdgroup handles one projection row
    // projection row index = byte_block * 8 + simd_group
    uint proj_row = byte_block * 8 + simd_group;
    if (proj_row >= params.sketch_dim) return;

    device const float* proj_ptr = projection + proj_row * D;

    // Dot product: S[proj_row] · key, excluding outlier channels
    // Tiled across SIMD lanes
    float acc = 0.0f;
    for (uint col = simd_lane; col < D; col += SIMD_WIDTH) {
        float mask_val = (outlier_mask[col] == 0) ? 1.0f : 0.0f;
        acc += proj_ptr[col] * shared_key[col] * mask_val;
    }

    // SIMD reduction
    acc = simd_sum(acc);

    // Sign bit: 1 if >= 0, 0 if < 0
    uint8_t sign_bit = (acc >= 0.0f) ? 1 : 0;

    // Step 3: Pack 8 sign bits from 8 simdgroups into 1 uint8
    // Use threadgroup memory for bit collection
    threadgroup uint8_t bit_staging[8];

    if (simd_lane == 0) {
        bit_staging[simd_group] = sign_bit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup 0, lane 0 does the packing
    if (simd_group == 0 && simd_lane == 0) {
        uint8_t packed = 0;
        for (uint b = 0; b < 8; b++) {
            packed |= (bit_staging[b] << (7 - b));
        }
        packed_bits[token_idx * bytes_per_token + byte_block] = packed;
    }

    // Step 4: Compute norm (only once per token, by simdgroup 0)
    if (byte_block == 0 && simd_group == 0) {
        float norm_acc = 0.0f;
        for (uint col = simd_lane; col < D; col += SIMD_WIDTH) {
            float v = shared_key[col];
            norm_acc += v * v;
        }
        norm_acc = simd_sum(norm_acc);
        if (simd_lane == 0) {
            norms[token_idx] = sqrt(norm_acc);
        }
    }
}


// ──────────────────────────────────────────────────────────────────────
// qjl_score: Asymmetric attention scoring
//   Full-precision query × quantized keys → attention logits
//
// Grid:  (B * H, Nq, ceil(Nk / SCORE_TOKENS_PER_TG))
// Threadgroup: (SIMD_WIDTH, SCORE_SIMDGROUPS)
//
// Each threadgroup computes scores for one query against a tile of keys.
// Step 1: Compute query_sketch = q @ S^T (shared across all key tokens)
// Step 2: For each quantized key, unpack bits and dot with query_sketch
// ──────────────────────────────────────────────────────────────────────

constant uint SCORE_SIMDGROUPS = 8;
constant uint SCORE_TOKENS_PER_TG = 32;  // keys processed per threadgroup

kernel void qjl_score(
    device const float* queries       [[buffer(0)]],   // (B, H, Nq, D)
    device const float* projection    [[buffer(1)]],   // (sketch_dim, embed_dim)
    device const uint8_t* packed_bits [[buffer(2)]],   // (B, H, Nk, sketch_dim/8)
    device const float* key_norms     [[buffer(3)]],   // (B, H, Nk)
    device float* scores              [[buffer(4)]],   // (B, H, Nq, Nk)
    constant QJLQuantParams& params   [[buffer(5)]],
    uint3 tg_pos    [[threadgroup_position_in_grid]],
    uint3 tid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint bh_idx = tg_pos.x;     // batch * head index
    uint q_idx = tg_pos.y;      // query token index
    uint k_tile = tg_pos.z;     // key token tile

    uint B = params.batch_size;
    uint H = params.num_heads;
    uint Nk = params.seq_len;
    uint D = params.embed_dim;
    uint S = params.sketch_dim;
    uint bytes_per_token = S / 8;

    if (bh_idx >= B * H) return;

    // Query pointer
    device const float* q_ptr = queries + (bh_idx * params.seq_len + q_idx) * D;
    // Note: for decode, Nq is typically 1, so q_idx should be < actual Nq
    // The caller must set grid.y = actual Nq

    // ── Step 1: Compute query_sketch = q @ S^T ──
    // Each simdgroup computes a tile of the sketch
    // sketch[i] = Σ_j q[j] * S[i][j]

    threadgroup float shared_query[1024];
    threadgroup float shared_sketch[1024];  // max sketch_dim = 1024

    // Load query into shared memory
    for (uint i = simd_group * SIMD_WIDTH + simd_lane; i < D;
         i += SCORE_SIMDGROUPS * SIMD_WIDTH) {
        shared_query[i] = q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute sketch: each thread computes partial dot products
    for (uint row = simd_group; row < S; row += SCORE_SIMDGROUPS) {
        device const float* proj_row = projection + row * D;
        float acc = 0.0f;
        for (uint col = simd_lane; col < D; col += SIMD_WIDTH) {
            acc += proj_row[col] * shared_query[col];
        }
        acc = simd_sum(acc);
        if (simd_lane == 0) {
            shared_sketch[row] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 2: Score each key token in this tile ──
    // Scale factor: sqrt(pi/2) / sketch_dim
    float scale = sqrt(M_PI_F / 2.0f) / float(S);

    uint k_start = k_tile * SCORE_TOKENS_PER_TG;

    // Each simdgroup handles different key tokens
    for (uint ki = k_start + simd_group; ki < min(k_start + SCORE_TOKENS_PER_TG, Nk);
         ki += SCORE_SIMDGROUPS) {

        device const uint8_t* bits = packed_bits + (bh_idx * Nk + ki) * bytes_per_token;
        float norm_k = key_norms[bh_idx * Nk + ki];

        // Unpack bits and compute inner product with query sketch
        // inner = Σ_i (bit_i ? +sketch_i : -sketch_i)
        float inner = 0.0f;

        // Each SIMD lane processes different bytes
        for (uint byte_idx = simd_lane; byte_idx < bytes_per_token; byte_idx += SIMD_WIDTH) {
            uint8_t byte_val = bits[byte_idx];
            uint base_bit = byte_idx * 8;

            // Unpack 8 bits and accumulate
            for (uint b = 0; b < 8; b++) {
                uint bit_pos = base_bit + b;
                if (bit_pos < S) {
                    float sketch_val = shared_sketch[bit_pos];
                    uint8_t bit = (byte_val >> (7 - b)) & 1;
                    inner += bit ? sketch_val : -sketch_val;
                }
            }
        }

        // SIMD reduction
        inner = simd_sum(inner);

        // Write score
        if (simd_lane == 0) {
            float score = scale * norm_k * inner;
            scores[bh_idx * params.seq_len * Nk + q_idx * Nk + ki] = score;
        }
    }
}
