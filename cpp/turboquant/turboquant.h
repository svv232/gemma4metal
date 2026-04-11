#pragma once

#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace turboquant {

using namespace mlx::core;

class PolarQuantize : public Primitive {
 public:
  PolarQuantize(Stream stream, int block_size)
      : Primitive(stream), block_size_(block_size) {}

  void eval_cpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;

  const char* name() const override { return "PolarQuantize"; }

  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PolarQuantize&>(other);
    return block_size_ == o.block_size_;
  }

 private:
  int block_size_;
};

class PolarDequantize : public Primitive {
 public:
  PolarDequantize(Stream stream, int block_size)
      : Primitive(stream), block_size_(block_size) {}

  void eval_cpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override;

  const char* name() const override { return "PolarDequantize"; }

  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PolarDequantize&>(other);
    return block_size_ == o.block_size_;
  }

 private:
  int block_size_;
};

std::vector<array> polar_quantize(
    const array& vectors,
    const array& precondition,
    const array& cb_l1,
    const array& cb_l2,
    const array& cb_l3,
    const array& cb_l4,
    int block_size = 16,
    StreamOrDevice s = {});

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
    int block_size = 16,
    StreamOrDevice s = {});

}  // namespace turboquant
