// TurboQuant MLX C++ Extension — CPU/GPU dispatch

#include "turboquant.h"
#include "mlx/backend/metal/device.h"
#include <cmath>

namespace turboquant {

// ──────────────────────────────────────────────────────────────────────
// CPU fallback (not implemented — GPU is the target)
// ────────────────��─────────────────────────────────���───────────────────

void PolarQuantize::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("PolarQuantize: use GPU (Metal)");
}

void PolarDequantize::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("PolarDequantize: use GPU (Metal)");
}

// ──────────────────────────────────────────────────────────────────────
// GPU: dispatch Metal kernels
// ────────────���───────────────────────────────────���─────────────────────

// Path to metallib — set at module load or via env
static std::string metallib_path;

void set_metallib_path(const std::string& path) {
  metallib_path = path;
}

void PolarQuantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // inputs: [vectors_preconditioned, cb_l1, cb_l2, cb_l3, cb_l4]
  auto& vectors = inputs[0];  // Already preconditioned (N_blocks * BS,) flat
  auto& cb_l1 = inputs[1];
  auto& cb_l2 = inputs[2];
  auto& cb_l3 = inputs[3];
  auto& cb_l4 = inputs[4];

  int total_elements = vectors.size();
  int N_blocks = total_elements / block_size_;

  // Allocate outputs
  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  // Get Metal device and command encoder
  auto& device = metal::device(stream().device);
  auto& encoder = device.get_command_encoder(stream().index);

  // Load metallib and get kernel
  auto* lib = device.get_library("turboquant", metallib_path);
  auto* kernel = device.get_kernel("polar_fused_quantize", lib);
  encoder.set_compute_pipeline_state(kernel);

  // Set buffers
  encoder.set_input_array(vectors, 0);
  encoder.set_input_array(cb_l1, 1);
  encoder.set_input_array(cb_l2, 2);
  encoder.set_input_array(cb_l3, 3);
  encoder.set_input_array(cb_l4, 4);

  // Codebook sizes as scalars
  uint32_t n1 = cb_l1.size(), n2 = cb_l2.size(), n3 = cb_l3.size(), n4 = cb_l4.size();
  encoder.set_bytes(n1, 5);
  encoder.set_bytes(n2, 6);
  encoder.set_bytes(n3, 7);
  encoder.set_bytes(n4, 8);

  // Output arrays
  encoder.set_output_array(outputs[0], 9);   // indices_l1
  encoder.set_output_array(outputs[1], 10);  // indices_l2
  encoder.set_output_array(outputs[2], 11);  // indices_l3
  encoder.set_output_array(outputs[3], 12);  // indices_l4
  encoder.set_output_array(outputs[4], 13);  // radii

  // Dispatch
  MTL::Size grid(N_blocks, 1, 1);
  MTL::Size threadgroup(1, 1, 1);
  encoder.dispatch_threadgroups(grid, threadgroup);
}

void PolarDequantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // inputs: [indices_l1..4, radii, cb_l1..4]
  auto& idx_l1 = inputs[0];
  auto& idx_l2 = inputs[1];
  auto& idx_l3 = inputs[2];
  auto& idx_l4 = inputs[3];
  auto& radii = inputs[4];
  auto& cb_l1 = inputs[5];
  auto& cb_l2 = inputs[6];
  auto& cb_l3 = inputs[7];
  auto& cb_l4 = inputs[8];

  int N_blocks = radii.size();

  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));

  auto& device = metal::device(stream().device);
  auto& encoder = device.get_command_encoder(stream().index);

  auto* lib = device.get_library("turboquant", metallib_path);
  auto* kernel = device.get_kernel("polar_inverse_dequantize", lib);
  encoder.set_compute_pipeline_state(kernel);

  encoder.set_input_array(cb_l1, 0);
  encoder.set_input_array(cb_l2, 1);
  encoder.set_input_array(cb_l3, 2);
  encoder.set_input_array(cb_l4, 3);
  encoder.set_input_array(idx_l1, 4);
  encoder.set_input_array(idx_l2, 5);
  encoder.set_input_array(idx_l3, 6);
  encoder.set_input_array(idx_l4, 7);
  encoder.set_input_array(radii, 8);
  encoder.set_output_array(outputs[0], 9);

  MTL::Size grid(N_blocks, 1, 1);
  MTL::Size threadgroup(1, 1, 1);
  encoder.dispatch_threadgroups(grid, threadgroup);
}

// ──────────────────────────────────────────────────────────────────────
// Python-facing functions
// ─────────────────────────────────────────────────���────────────────────

std::vector<array> polar_quantize(
    const array& vectors,
    const array& precondition,
    const array& cb_l1,
    const array& cb_l2,
    const array& cb_l3,
    const array& cb_l4,
    int block_size,
    StreamOrDevice s) {

  int N = vectors.shape(0);
  int D = vectors.shape(1);
  int n_blocks = D / block_size;

  // Precondition: y = vectors @ P^T (use MLX matmul)
  auto y = matmul(vectors, transpose(precondition));
  auto y_flat = reshape(y, {-1});

  // Create outputs
  auto p = std::make_shared<PolarQuantize>(to_stream(s), block_size);
  return array::make_arrays(
      {{N * n_blocks * (block_size / 2)},
       {N * n_blocks * (block_size / 4)},
       {N * n_blocks * (block_size / 8)},
       {N * n_blocks * (block_size / 16)},
       {N * n_blocks}},
      {uint8, uint8, uint8, uint8, float16},
      p,
      {y_flat, cb_l1, cb_l2, cb_l3, cb_l4});
}

array polar_dequantize(
    const array& indices_l1,
    const array& indices_l2,
    const array& indices_l3,
    const array& indices_l4,
    const array& radii,
    const array& precondition,
    const array& cb_l1,
    const array& cb_l2,
    const array& cb_l3,
    const array& cb_l4,
    int block_size,
    StreamOrDevice s) {

  int N_blocks = radii.size();
  int D = N_blocks * block_size;
  int N = indices_l1.size() / (N_blocks / (D / block_size) * (block_size / 2));

  auto p = std::make_shared<PolarDequantize>(to_stream(s), block_size);
  auto y_flat = array(
      {N_blocks * block_size},
      float32,
      p,
      {indices_l1, indices_l2, indices_l3, indices_l4, radii,
       cb_l1, cb_l2, cb_l3, cb_l4});

  // Inverse precondition: x = y @ P (P is orthogonal, so P^T = P^-1)
  auto y_reshaped = reshape(y_flat, {-1, D});
  return matmul(y_reshaped, precondition);
}

}  // namespace turboquant
