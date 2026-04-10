// QJL Quantization Kernel for Apple Silicon Metal
//
// Projects key vectors through random matrix S, takes sign bits, packs into uint8.
// Port of CUDA qjl_quant_kernel from github.com/amirzandieh/QJL
//
// Grid:  (batch * n_heads * seq_len, blocks_per_group, num_proj_blocks)
// Threadgroup: (32, 32) — 32 threads/simdgroup, 32 simdgroups
//
// TODO: Implement full kernel. This is the scaffold.

#include <metal_stdlib>
using namespace metal;

// Constants
constant uint SIMD_WIDTH = 32;
constant uint SIMDGROUPS_PER_TG = 32;

struct QJLQuantParams {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint embed_dim;
    uint sketch_dim;
    uint num_outliers;
};

kernel void qjl_quant(
    device const float* keys          [[buffer(0)]],   // (B, H, N, D)
    device const float* projection    [[buffer(1)]],   // (sketch_dim, embed_dim)
    device const uint8_t* outlier_mask [[buffer(2)]],  // (D,) — 1 if outlier
    device uint8_t* packed_bits       [[buffer(3)]],   // (B, H, N, sketch_dim/8)
    device float* norms               [[buffer(4)]],   // (B, H, N)
    constant QJLQuantParams& params   [[buffer(5)]],
    uint3 tg_pos    [[threadgroup_position_in_grid]],
    uint3 tid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // TODO: Full implementation
    // Step 1: Load outlier mask into threadgroup memory
    // Step 2: Load key vector tile (transpose for coalesced access)
    // Step 3: For each projection row:
    //   a. Dot product: accumulate S[row][col] * key[col] for inlier cols
    //   b. sign(accumulator)
    //   c. Pack 8 sign bits -> uint8 via shift-and-OR
    // Step 4: Write packed bits + compute norm
}

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
    // TODO: Full implementation
    // Step 1: Load query into threadgroup memory
    // Step 2: Compute query_sketch = q @ S^T (tiled)
    // Step 3: For each quantized key token:
    //   a. Load packed uint8, unpack: (byte >> shift) & 1
    //   b. Accumulate: bit ? +query_sketch_i : -query_sketch_i
    //   c. Simd reduce sum
    //   d. Scale: score = sqrt(pi/2) / sketch_dim * norm_k * sum
    // Step 4: Write attention logit
}
