// TurboQuant fused Metal kernel: forward_polar + codebook_nearest
//
// Compiled into metallib at build time, loaded by eval_gpu.

#include <metal_stdlib>
using namespace metal;

kernel void polar_fused_quantize(
    device const float* input        [[buffer(0)]],   // (N_blocks * block_size)
    device const float* cb_l1        [[buffer(1)]],   // level 1 codebook
    device const float* cb_l2        [[buffer(2)]],   // level 2 codebook
    device const float* cb_l3        [[buffer(3)]],   // level 3 codebook
    device const float* cb_l4        [[buffer(4)]],   // level 4 codebook
    device const uint& n_cb1         [[buffer(5)]],   // codebook sizes
    device const uint& n_cb2         [[buffer(6)]],
    device const uint& n_cb3         [[buffer(7)]],
    device const uint& n_cb4         [[buffer(8)]],
    device uint8_t* indices_l1       [[buffer(9)]],
    device uint8_t* indices_l2       [[buffer(10)]],
    device uint8_t* indices_l3       [[buffer(11)]],
    device uint8_t* indices_l4       [[buffer(12)]],
    device half* radii               [[buffer(13)]],
    uint block_idx                   [[thread_position_in_grid]]
) {
    float r[16];
    for (uint j = 0; j < 16; j++) {
        r[j] = input[block_idx * 16 + j];
    }

    // Level 1: 16 → 8
    {
        float tmp[8];
        for (uint j = 0; j < 8; j++) {
            float a = r[2*j], b = r[2*j+1];
            float angle = metal::atan2(b, a);
            if (angle < 0.0f) angle += 2.0f * M_PI_F;
            float best = 1e10f;
            uint8_t idx = 0;
            for (uint c = 0; c < n_cb1; c++) {
                float d = metal::abs(angle - cb_l1[c]);
                if (d < best) { best = d; idx = c; }
            }
            indices_l1[block_idx * 8 + j] = idx;
            tmp[j] = metal::sqrt(a*a + b*b);
        }
        for (uint j = 0; j < 8; j++) r[j] = tmp[j];
    }

    // Level 2: 8 → 4
    {
        float tmp[4];
        for (uint j = 0; j < 4; j++) {
            float a = r[2*j], b = r[2*j+1];
            float angle = metal::atan2(b, a);
            float best = 1e10f;
            uint8_t idx = 0;
            for (uint c = 0; c < n_cb2; c++) {
                float d = metal::abs(angle - cb_l2[c]);
                if (d < best) { best = d; idx = c; }
            }
            indices_l2[block_idx * 4 + j] = idx;
            tmp[j] = metal::sqrt(a*a + b*b);
        }
        for (uint j = 0; j < 4; j++) r[j] = tmp[j];
    }

    // Level 3: 4 → 2
    {
        float tmp[2];
        for (uint j = 0; j < 2; j++) {
            float a = r[2*j], b = r[2*j+1];
            float angle = metal::atan2(b, a);
            float best = 1e10f;
            uint8_t idx = 0;
            for (uint c = 0; c < n_cb3; c++) {
                float d = metal::abs(angle - cb_l3[c]);
                if (d < best) { best = d; idx = c; }
            }
            indices_l3[block_idx * 2 + j] = idx;
            tmp[j] = metal::sqrt(a*a + b*b);
        }
        r[0] = tmp[0]; r[1] = tmp[1];
    }

    // Level 4: 2 → 1
    {
        float a = r[0], b = r[1];
        float angle = metal::atan2(b, a);
        float best = 1e10f;
        uint8_t idx = 0;
        for (uint c = 0; c < n_cb4; c++) {
            float d = metal::abs(angle - cb_l4[c]);
            if (d < best) { best = d; idx = c; }
        }
        indices_l4[block_idx] = idx;
        radii[block_idx] = static_cast<half>(metal::sqrt(a*a + b*b));
    }
}

kernel void polar_inverse_dequantize(
    device const float* cb_l1        [[buffer(0)]],
    device const float* cb_l2        [[buffer(1)]],
    device const float* cb_l3        [[buffer(2)]],
    device const float* cb_l4        [[buffer(3)]],
    device const uint8_t* indices_l1 [[buffer(4)]],
    device const uint8_t* indices_l2 [[buffer(5)]],
    device const uint8_t* indices_l3 [[buffer(6)]],
    device const uint8_t* indices_l4 [[buffer(7)]],
    device const half* radii         [[buffer(8)]],
    device float* output             [[buffer(9)]],
    uint block_idx                   [[thread_position_in_grid]]
) {
    float r[16];
    r[0] = static_cast<float>(radii[block_idx]);

    // Level 4: 1 → 2
    {
        uint idx = indices_l4[block_idx];
        float theta = cb_l4[idx];
        float r0 = r[0];
        r[0] = r0 * metal::cos(theta);
        r[1] = r0 * metal::sin(theta);
    }

    // Level 3: 2 → 4
    {
        float tmp[4];
        for (uint j = 0; j < 2; j++) {
            uint idx = indices_l3[block_idx * 2 + j];
            float theta = cb_l3[idx];
            tmp[2*j]   = r[j] * metal::cos(theta);
            tmp[2*j+1] = r[j] * metal::sin(theta);
        }
        for (uint j = 0; j < 4; j++) r[j] = tmp[j];
    }

    // Level 2: 4 → 8
    {
        float tmp[8];
        for (uint j = 0; j < 4; j++) {
            uint idx = indices_l2[block_idx * 4 + j];
            float theta = cb_l2[idx];
            tmp[2*j]   = r[j] * metal::cos(theta);
            tmp[2*j+1] = r[j] * metal::sin(theta);
        }
        for (uint j = 0; j < 8; j++) r[j] = tmp[j];
    }

    // Level 1: 8 → 16
    {
        float tmp[16];
        for (uint j = 0; j < 8; j++) {
            uint idx = indices_l1[block_idx * 8 + j];
            float theta = cb_l1[idx];
            tmp[2*j]   = r[j] * metal::cos(theta);
            tmp[2*j+1] = r[j] * metal::sin(theta);
        }
        for (uint j = 0; j < 16; j++) r[j] = tmp[j];
    }

    for (uint j = 0; j < 16; j++) {
        output[block_idx * 16 + j] = r[j];
    }
}
