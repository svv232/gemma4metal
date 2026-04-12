// Python bindings using nb::object to bypass type caster issue
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "turboquant.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

namespace turboquant { void set_metallib_path(const std::string& path); }

// Helper: extract mlx::core::array from a Python nb::object
// Uses nanobind's internal type casting via nb::cast
static array to_array(nb::handle obj) {
  return nb::cast<array>(obj);
}

NB_MODULE(turboquant_ext, m) {
  m.doc() = "TurboQuant: Fused PolarQuant KV cache compression for MLX";

  nb::module_::import_("mlx.core");

  m.def("set_metallib_path", &turboquant::set_metallib_path, "path"_a);

  m.def(
      "polar_quantize",
      [](nb::object vectors, nb::object precondition,
         nb::object cb_l1, nb::object cb_l2, nb::object cb_l3, nb::object cb_l4,
         int block_size) -> nb::list {
        auto result = turboquant::polar_quantize(
            to_array(vectors), to_array(precondition),
            to_array(cb_l1), to_array(cb_l2), to_array(cb_l3), to_array(cb_l4),
            block_size);
        nb::list out;
        for (auto& a : result) {
          out.append(nb::cast(a));
        }
        return out;
      },
      "vectors"_a, "precondition"_a,
      "cb_l1"_a, "cb_l2"_a, "cb_l3"_a, "cb_l4"_a,
      "block_size"_a = 16);

  m.def(
      "polar_dequantize",
      [](nb::object idx_l1, nb::object idx_l2, nb::object idx_l3, nb::object idx_l4,
         nb::object radii, nb::object precondition,
         nb::object cb_l1, nb::object cb_l2, nb::object cb_l3, nb::object cb_l4,
         int block_size) -> nb::object {
        auto result = turboquant::polar_dequantize(
            to_array(idx_l1), to_array(idx_l2), to_array(idx_l3), to_array(idx_l4),
            to_array(radii), to_array(precondition),
            to_array(cb_l1), to_array(cb_l2), to_array(cb_l3), to_array(cb_l4),
            block_size);
        return nb::cast(result);
      },
      "idx_l1"_a, "idx_l2"_a, "idx_l3"_a, "idx_l4"_a,
      "radii"_a, "precondition"_a,
      "cb_l1"_a, "cb_l2"_a, "cb_l3"_a, "cb_l4"_a,
      "block_size"_a = 16);
}
