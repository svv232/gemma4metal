// PolarQuant Transform Kernels for Apple Silicon Metal
//
// Implements recursive polar coordinate transform:
//   Forward: Cartesian → (angles per level, radius)
//   Inverse: (quantized angles, radius) → Cartesian
//
// Block size 16 → 4 levels, all fits in registers.
// Codebooks are tiny (max 16 entries per level) → passed as input buffers.
//
// Implemented via MLX custom Metal kernels (mx.fast.metal_kernel).
// This file documents the standalone Metal shader interface for reference.
// The actual dispatch is in python/polar_metal.py.

#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_SIZE = 16;
constant uint N_LEVELS = 4;  // log2(BLOCK_SIZE)

struct PolarParams {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint embed_dim;
    uint block_size;
    uint n_levels;
    uint bits_per_level[4];      // [4, 2, 2, 2] for default config
    uint codebook_sizes[4];      // [16, 4, 4, 4]
    uint codebook_offsets[4];    // offsets into flat codebook array
};

// ──────────────────────────────────────────────────────────────────────
// Forward polar transform
// Each thread processes one block of BLOCK_SIZE coordinates.
// ──────────────────────────────────────────────────────────────────────
kernel void polar_forward(
    device const float* input         [[buffer(0)]],   // (N_blocks * 16)
    device float* angles_l1           [[buffer(1)]],   // (N_blocks * 8)
    device float* angles_l2           [[buffer(2)]],   // (N_blocks * 4)
    device float* angles_l3           [[buffer(3)]],   // (N_blocks * 2)
    device float* angles_l4           [[buffer(4)]],   // (N_blocks)
    device float* radii               [[buffer(5)]],   // (N_blocks)
    uint tid [[thread_position_in_grid]]
) {
    uint block_idx = tid;

    float r[16];
    for (uint j = 0; j < 16; j++) {
        r[j] = input[block_idx * 16 + j];
    }

    // Level 1: 16 → 8, angles in [0, 2π)
    {
        float tmp[8];
        for (uint j = 0; j < 8; j++) {
            float a = r[2*j], b = r[2*j+1];
            float angle = atan2(b, a);
            if (angle < 0.0f) angle += 2.0f * M_PI_F;
            angles_l1[block_idx * 8 + j] = angle;
            tmp[j] = sqrt(a*a + b*b);
        }
        for (uint j = 0; j < 8; j++) r[j] = tmp[j];
    }

    // Level 2: 8 → 4
    {
        float tmp[4];
        for (uint j = 0; j < 4; j++) {
            float a = r[2*j], b = r[2*j+1];
            angles_l2[block_idx * 4 + j] = atan2(b, a);
            tmp[j] = sqrt(a*a + b*b);
        }
        for (uint j = 0; j < 4; j++) r[j] = tmp[j];
    }

    // Level 3: 4 → 2
    {
        float tmp[2];
        for (uint j = 0; j < 2; j++) {
            float a = r[2*j], b = r[2*j+1];
            angles_l3[block_idx * 2 + j] = atan2(b, a);
            tmp[j] = sqrt(a*a + b*b);
        }
        r[0] = tmp[0]; r[1] = tmp[1];
    }

    // Level 4: 2 → 1
    {
        float a = r[0], b = r[1];
        angles_l4[block_idx] = atan2(b, a);
        radii[block_idx] = sqrt(a*a + b*b);
    }
}

// ──────────────────────────────────────────────────────────────────────
// Inverse polar transform (dequantization)
// Each thread reconstructs one block of BLOCK_SIZE coordinates
// from quantized angle indices + radius.
// ──────────────────────────────────────────────────────────────────────
kernel void polar_inverse(
    device const float* codebook_l1   [[buffer(0)]],   // (n_centroids_l1)
    device const float* codebook_l2   [[buffer(1)]],   // (n_centroids_l2)
    device const float* codebook_l3   [[buffer(2)]],   // (n_centroids_l3)
    device const float* codebook_l4   [[buffer(3)]],   // (n_centroids_l4)
    device const uint* indices_l1     [[buffer(4)]],   // (N_blocks * 8)
    device const uint* indices_l2     [[buffer(5)]],   // (N_blocks * 4)
    device const uint* indices_l3     [[buffer(6)]],   // (N_blocks * 2)
    device const uint* indices_l4     [[buffer(7)]],   // (N_blocks)
    device const float* radii         [[buffer(8)]],   // (N_blocks)
    device float* output              [[buffer(9)]],   // (N_blocks * 16)
    uint tid [[thread_position_in_grid]]
) {
    uint block_idx = tid;

    float r[16];
    r[0] = radii[block_idx];

    // Level 4 → 2 values
    {
        uint idx = indices_l4[block_idx];
        float theta = codebook_l4[idx];
        float r0 = r[0];
        r[0] = r0 * cos(theta);
        r[1] = r0 * sin(theta);
    }

    // Level 3 → 4 values
    {
        float tmp[4];
        for (uint j = 0; j < 2; j++) {
            uint idx = indices_l3[block_idx * 2 + j];
            float theta = codebook_l3[idx];
            tmp[2*j]   = r[j] * cos(theta);
            tmp[2*j+1] = r[j] * sin(theta);
        }
        for (uint j = 0; j < 4; j++) r[j] = tmp[j];
    }

    // Level 2 → 8 values
    {
        float tmp[8];
        for (uint j = 0; j < 4; j++) {
            uint idx = indices_l2[block_idx * 4 + j];
            float theta = codebook_l2[idx];
            tmp[2*j]   = r[j] * cos(theta);
            tmp[2*j+1] = r[j] * sin(theta);
        }
        for (uint j = 0; j < 8; j++) r[j] = tmp[j];
    }

    // Level 1 → 16 values
    {
        float tmp[16];
        for (uint j = 0; j < 8; j++) {
            uint idx = indices_l1[block_idx * 8 + j];
            float theta = codebook_l1[idx];
            tmp[2*j]   = r[j] * cos(theta);
            tmp[2*j+1] = r[j] * sin(theta);
        }
        for (uint j = 0; j < 16; j++) r[j] = tmp[j];
    }

    for (uint j = 0; j < 16; j++) {
        output[block_idx * 16 + j] = r[j];
    }
}
