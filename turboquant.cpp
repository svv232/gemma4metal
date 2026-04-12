// TurboQuant MLX C++ Extension with Metal dispatch
#include "turboquant.h"
#include "mlx/backend/metal/device.h"
#include <cmath>

namespace turboquant {

static std::string metallib_path;
void set_metallib_path(const std::string& path) { metallib_path = path; }

void PolarQuantize::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("PolarQuantize: use GPU");
}

void PolarQuantize::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& vectors = inputs[0];
  auto& cb_l1 = inputs[1];
  auto& cb_l2 = inputs[2];
  auto& cb_l3 = inputs[3];
  auto& cb_l4 = inputs[4];

  int total = vectors.size();
  int N_blocks = total / block_size_;

  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  auto& device = metal::device(stream().device);
  auto& encoder = device.get_command_encoder(stream().index);

  auto* lib = device.get_library("turboquant", metallib_path);
  auto* kernel = device.get_kernel("polar_fused_quantize", lib);
  encoder.set_compute_pipeline_state(kernel);

  encoder.set_input_array(vectors, 0);
  encoder.set_input_array(cb_l1, 1);
  encoder.set_input_array(cb_l2, 2);
  encoder.set_input_array(cb_l3, 3);
  encoder.set_input_array(cb_l4, 4);

  uint32_t n1 = cb_l1.size(), n2 = cb_l2.size(), n3 = cb_l3.size(), n4 = cb_l4.size();
  encoder.set_bytes(n1, 5);
  encoder.set_bytes(n2, 6);
  encoder.set_bytes(n3, 7);
  encoder.set_bytes(n4, 8);

  encoder.set_output_array(outputs[0], 9);
  encoder.set_output_array(outputs[1], 10);
  encoder.set_output_array(outputs[2], 11);
  encoder.set_output_array(outputs[3], 12);
  encoder.set_output_array(outputs[4], 13);

  MTL::Size grid(N_blocks, 1, 1);
  MTL::Size threadgroup(1, 1, 1);
  encoder.dispatch_threadgroups(grid, threadgroup);
}

void PolarDequantize::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("PolarDequantize: use GPU");
}

void PolarDequantize::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  // inputs: {i1, i2, i3, i4, radii, cb1, cb2, cb3, cb4}
  //          0    1   2   3   4      5    6    7    8
  // Metal kernel expects: cb1(0), cb2(1), cb3(2), cb4(3), i1(4), i2(5), i3(6), i4(7), radii(8), out(9)
  auto& radii = inputs[4];
  int N_blocks = radii.size();

  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));

  auto& device = metal::device(stream().device);
  auto& encoder = device.get_command_encoder(stream().index);

  auto* lib = device.get_library("turboquant", metallib_path);
  auto* kernel = device.get_kernel("polar_inverse_dequantize", lib);
  encoder.set_compute_pipeline_state(kernel);

  // Map inputs to Metal buffer indices
  encoder.set_input_array(inputs[5], 0);  // cb_l1
  encoder.set_input_array(inputs[6], 1);  // cb_l2
  encoder.set_input_array(inputs[7], 2);  // cb_l3
  encoder.set_input_array(inputs[8], 3);  // cb_l4
  encoder.set_input_array(inputs[0], 4);  // indices_l1
  encoder.set_input_array(inputs[1], 5);  // indices_l2
  encoder.set_input_array(inputs[2], 6);  // indices_l3
  encoder.set_input_array(inputs[3], 7);  // indices_l4
  encoder.set_input_array(inputs[4], 8);  // radii
  encoder.set_output_array(outputs[0], 9);

  MTL::Size grid(N_blocks, 1, 1);
  MTL::Size threadgroup(1, 1, 1);
  encoder.dispatch_threadgroups(grid, threadgroup);
}

// Python-facing functions
std::vector<array> polar_quantize(
    const array& vectors, const array& precondition,
    const array& cb_l1, const array& cb_l2, const array& cb_l3, const array& cb_l4,
    int block_size, StreamOrDevice s) {
  int N = vectors.shape(0), D = vectors.shape(1), nb = D / block_size;
  auto y = matmul(vectors, transpose(precondition));
  auto y_flat = reshape(y, {-1});
  auto p = std::make_shared<PolarQuantize>(to_stream(s), block_size);
  return array::make_arrays(
      {{N*nb*(block_size/2)}, {N*nb*(block_size/4)}, {N*nb*(block_size/8)},
       {N*nb*(block_size/16)}, {N*nb}},
      {uint8, uint8, uint8, uint8, float16},
      p, {y_flat, cb_l1, cb_l2, cb_l3, cb_l4});
}

array polar_dequantize(
    const array& i1, const array& i2, const array& i3, const array& i4,
    const array& radii, const array& precondition,
    const array& cb1, const array& cb2, const array& cb3, const array& cb4,
    int block_size, StreamOrDevice s) {
  int N_blocks = radii.size();
  int D = precondition.shape(0);  // D from precondition matrix shape
  auto p = std::make_shared<PolarDequantize>(to_stream(s), block_size);
  auto y = array({N_blocks * block_size}, float32, p,
      {i1, i2, i3, i4, radii, cb1, cb2, cb3, cb4});
  return matmul(reshape(y, {-1, D}), precondition);
}

// Compressed SDPA
void CompressedSDPA::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("CompressedSDPA: use GPU");
}

void CompressedSDPA::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  // inputs: {queries, idx_l1..l4, radii, cb_l1..l4, values, params}
  // The Metal kernel sdpa_turboquant handles everything
  auto& queries = inputs[0];
  auto& radii = inputs[5];
  auto& values = inputs[10];
  int Nq = queries.shape(0);
  int D_ = queries.shape(1);

  outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));

  auto& device = metal::device(stream().device);
  auto& encoder = device.get_command_encoder(stream().index);

  auto* lib = device.get_library("turboquant", metallib_path);
  auto* kernel = device.get_kernel("sdpa_tq_sequential", lib);
  encoder.set_compute_pipeline_state(kernel);

  // Set buffers matching sdpa_tq_sequential signature
  encoder.set_input_array(inputs[0], 0);   // queries
  encoder.set_input_array(inputs[1], 1);   // idx_l1
  encoder.set_input_array(inputs[2], 2);   // idx_l2
  encoder.set_input_array(inputs[3], 3);   // idx_l3
  encoder.set_input_array(inputs[4], 4);   // idx_l4
  encoder.set_input_array(inputs[5], 5);   // radii
  encoder.set_input_array(inputs[6], 6);   // cb1
  encoder.set_input_array(inputs[7], 7);   // cb2
  encoder.set_input_array(inputs[8], 8);   // cb3
  encoder.set_input_array(inputs[9], 9);   // cb4
  encoder.set_input_array(inputs[10], 10); // values
  encoder.set_output_array(outputs[0], 11);
  encoder.set_input_array(inputs[11], 12); // params

  // 24 simdgroups × 32 lanes = 768 threads per head
  int n_heads = Nq;
  MTL::Size grid(n_heads, 1, 1);
  MTL::Size tg(768, 1, 1);  // 24 simdgroups
  encoder.dispatch_threadgroups(grid, tg);
}

array compressed_sdpa(
    const array& queries, const array& i1, const array& i2, const array& i3, const array& i4,
    const array& radii, const array& cb1, const array& cb2, const array& cb3, const array& cb4,
    const array& values, int block_size, float scale, StreamOrDevice s) {

  int Nq = queries.shape(0);
  int D_ = queries.shape(1);
  int N = values.shape(0);
  int nb = D_ / block_size;

  // Build params struct
  struct { int N; int D; int nb; float scale; int gqa; } params_data = {N, D_, nb, scale, 1};
  auto params = array(reinterpret_cast<const uint8_t*>(&params_data), {static_cast<int>(sizeof(params_data))}, uint8);

  auto p = std::make_shared<CompressedSDPA>(to_stream(s), block_size, scale);
  return array(
      {Nq, D_}, float32, p,
      {queries, i1, i2, i3, i4, radii, cb1, cb2, cb3, cb4, values, params});
}

// Fast dequantize: inverse polar only, no precondition matmul
array polar_dequantize_fast(
    const array& i1, const array& i2, const array& i3, const array& i4,
    const array& radii, const array& cb1, const array& cb2, const array& cb3, const array& cb4,
    int D, int block_size, StreamOrDevice s) {
  int N_blocks = radii.size();
  auto p = std::make_shared<PolarDequantize>(to_stream(s), block_size);
  auto y = array({N_blocks * block_size}, float32, p,
      {i1, i2, i3, i4, radii, cb1, cb2, cb3, cb4});
  return reshape(y, {-1, D});  // Just reshape, no matmul!
}

// Packed quantize/dequant primitives
class PackedPolarQuantize : public Primitive {
 public:
  PackedPolarQuantize(Stream stream, int bs) : Primitive(stream), block_size_(bs) {}
  void eval_cpu(const std::vector<array>& i, std::vector<array>& o) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& vectors = inputs[0];
    int N_blocks = vectors.size() / block_size_;
    for (auto& out : outputs) out.set_data(allocator::malloc(out.nbytes()));
    auto& dev = metal::device(stream().device);
    auto& enc = dev.get_command_encoder(stream().index);
    auto* lib = dev.get_library("turboquant", metallib_path);
    auto* kern = dev.get_kernel("polar_quantize_packed", lib);
    enc.set_compute_pipeline_state(kern);
    for (int i = 0; i < 5; i++) enc.set_input_array(inputs[i], i);
    uint32_t n1=inputs[1].size(), n2=inputs[2].size(), n3=inputs[3].size(), n4=inputs[4].size();
    enc.set_bytes(n1, 5); enc.set_bytes(n2, 6); enc.set_bytes(n3, 7); enc.set_bytes(n4, 8);
    enc.set_output_array(outputs[0], 9);  // packed_l1 (uint32)
    enc.set_output_array(outputs[1], 10); // packed_l2 (uint8)
    enc.set_output_array(outputs[2], 11); // packed_l3 (uint8)
    enc.set_output_array(outputs[3], 12); // packed_l4 (uint8)
    enc.set_output_array(outputs[4], 13); // radii (float16)
    enc.dispatch_threadgroups(MTL::Size(N_blocks, 1, 1), MTL::Size(1, 1, 1));
  }
  const char* name() const override { return "PackedPolarQuantize"; }
  bool is_equivalent(const Primitive& other) const override { return true; }
 private:
  int block_size_;
};

class PackedPolarDequantize : public Primitive {
 public:
  PackedPolarDequantize(Stream stream, int bs) : Primitive(stream), block_size_(bs) {}
  void eval_cpu(const std::vector<array>& i, std::vector<array>& o) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    int N_blocks = inputs[4].size();  // radii
    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
    auto& dev = metal::device(stream().device);
    auto& enc = dev.get_command_encoder(stream().index);
    auto* lib = dev.get_library("turboquant", metallib_path);
    auto* kern = dev.get_kernel("polar_inverse_packed", lib);
    enc.set_compute_pipeline_state(kern);
    // Codebooks: inputs 0-3, packed indices: inputs 4-7, radii: input 8
    for (int i = 0; i < 9; i++) enc.set_input_array(inputs[i], i);
    enc.set_output_array(outputs[0], 9);
    enc.dispatch_threadgroups(MTL::Size(N_blocks, 1, 1), MTL::Size(1, 1, 1));
  }
  const char* name() const override { return "PackedPolarDequantize"; }
  bool is_equivalent(const Primitive& other) const override { return true; }
 private:
  int block_size_;
};

std::vector<array> polar_quantize_packed(
    const array& vectors, const array& precondition,
    const array& cb1, const array& cb2, const array& cb3, const array& cb4,
    int block_size, StreamOrDevice s) {
  int N = vectors.shape(0), D = vectors.shape(1), nb = D / block_size;
  auto y = matmul(vectors, transpose(precondition));
  auto y_flat = reshape(y, {-1});
  int N_blocks = N * nb;
  auto p = std::make_shared<PackedPolarQuantize>(to_stream(s), block_size);
  return array::make_arrays(
      {{N_blocks},     // packed_l1 (1 uint32 per block)
       {N_blocks},     // packed_l2 (1 uint8 per block)
       {N_blocks},     // packed_l3 (1 uint8 per block)
       {N_blocks},     // packed_l4 (1 uint8 per block)
       {N_blocks}},    // radii (float16)
      {uint32, uint8, uint8, uint8, float16},
      p, {y_flat, cb1, cb2, cb3, cb4});
}

array polar_dequantize_packed(
    const array& p1, const array& p2, const array& p3, const array& p4,
    const array& radii, const array& cb1, const array& cb2, const array& cb3, const array& cb4,
    int D, int block_size, StreamOrDevice s) {
  int N_blocks = radii.size();
  auto p = std::make_shared<PackedPolarDequantize>(to_stream(s), block_size);
  auto y = array({N_blocks * block_size}, float32, p,
      {cb1, cb2, cb3, cb4, p1, p2, p3, p4, radii});
  return reshape(y, {-1, D});
}

// Fast packed dequant with precomputed cos/sin (zero trig)
class PackedPolarDequantizeFast : public Primitive {
 public:
  PackedPolarDequantizeFast(Stream stream, int bs) : Primitive(stream), bs_(bs) {}
  void eval_cpu(const std::vector<array>& i, std::vector<array>& o) override {
    throw std::runtime_error("GPU only");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    int N_blocks = inputs[8].size();
    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
    auto& dev = metal::device(stream().device);
    auto& enc = dev.get_command_encoder(stream().index);
    auto* lib = dev.get_library("turboquant", metallib_path);
    auto* kern = dev.get_kernel("polar_inverse_packed_fast", lib);
    enc.set_compute_pipeline_state(kern);
    for (int i = 0; i < 9; i++) enc.set_input_array(inputs[i], i);
    enc.set_output_array(outputs[0], 9);
    enc.dispatch_threadgroups(MTL::Size(N_blocks, 1, 1), MTL::Size(1, 1, 1));
  }
  const char* name() const override { return "PackedDequantFast"; }
  bool is_equivalent(const Primitive& other) const override { return true; }
 private:
  int bs_;
};

array polar_dequantize_packed_fast(
    const array& p1, const array& p2, const array& p3, const array& p4,
    const array& radii,
    const array& cs1, const array& cs2, const array& cs3, const array& cs4,
    int D, int block_size, StreamOrDevice s) {
  int N_blocks = radii.size();
  auto p = std::make_shared<PackedPolarDequantizeFast>(to_stream(s), block_size);
  auto y = array({N_blocks * block_size}, float32, p,
      {cs1, cs2, cs3, cs4, p1, p2, p3, p4, radii});
  return reshape(y, {-1, D});
}

// Compressed scoring primitive
class CompressedScore : public Primitive {
 public:
  CompressedScore(Stream stream, int bs, float sc, int n)
      : Primitive(stream), block_size_(bs), scale_(sc), N_(n) {}
  void eval_cpu(const std::vector<array>& i, std::vector<array>& o) override {
    throw std::runtime_error("use GPU");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
    auto& dev = metal::device(stream().device);
    auto& enc = dev.get_command_encoder(stream().index);
    auto* lib = dev.get_library("turboquant", metallib_path);
    auto* kern = dev.get_kernel("tq_score_shared", lib);
    enc.set_compute_pipeline_state(kern);
    for (int i = 0; i < 10; i++) enc.set_input_array(inputs[i], i);
    enc.set_output_array(outputs[0], 10);
    enc.set_input_array(inputs[10], 11);
    int Nq = inputs[0].shape(0);
    enc.dispatch_threadgroups(MTL::Size(Nq, 1, 1), MTL::Size(768, 1, 1));
  }
  const char* name() const override { return "CompressedScore"; }
  bool is_equivalent(const Primitive& other) const override { return true; }
 private:
  int block_size_, N_;
  float scale_;
};

array compressed_score(
    const array& queries, const array& i1, const array& i2, const array& i3, const array& i4,
    const array& radii, const array& cb1, const array& cb2, const array& cb3, const array& cb4,
    int N, int block_size, float scale, StreamOrDevice s) {
  int Nq = queries.shape(0), D_ = queries.shape(1), nb = D_ / block_size;
  struct { uint32_t N, D, NB; float scale; } pd = {(uint32_t)N, (uint32_t)D_, (uint32_t)nb, scale};
  auto params = array(reinterpret_cast<const uint8_t*>(&pd), {(int)sizeof(pd)}, uint8);
  auto p = std::make_shared<CompressedScore>(to_stream(s), block_size, scale, N);
  return array({Nq, N}, float32, p, {queries, i1, i2, i3, i4, radii, cb1, cb2, cb3, cb4, params});
}

}  // namespace turboquant
