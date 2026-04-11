// Python bindings for TurboQuant MLX extension
// Uses nanobind (MLX's preferred binding library)

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "mlx/mlx.h"
#include "turboquant.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;
using namespace turboquant;

// Forward declare
namespace turboquant { void set_metallib_path(const std::string& path); }

NB_MODULE(turboquant_ext, m) {
  m.doc() = "TurboQuant: Fused PolarQuant KV cache compression for MLX";

  // Import mlx.core to get access to array type casters
  nb::module_::import_("mlx.core");

  m.def("set_metallib_path", &turboquant::set_metallib_path, "path"_a,
        "Set path to turboquant.metallib");

  m.def(
      "polar_quantize",
      &polar_quantize,
      "vectors"_a,
      "precondition"_a,
      "cb_l1"_a,
      "cb_l2"_a,
      "cb_l3"_a,
      "cb_l4"_a,
      "block_size"_a = 16,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused PolarQuant quantize: precondition → forward_polar → codebook_nearest.

        Args:
            vectors: (N, D) input vectors
            precondition: (D, D) orthogonal preconditioning matrix
            cb_l1..cb_l4: codebook arrays for each polar level
            block_size: polar block size (default 16)

        Returns:
            List of [indices_l1, indices_l2, indices_l3, indices_l4, radii]
      )");

  m.def(
      "polar_dequantize",
      &polar_dequantize,
      "indices_l1"_a,
      "indices_l2"_a,
      "indices_l3"_a,
      "indices_l4"_a,
      "radii"_a,
      "precondition"_a,
      "cb_l1"_a,
      "cb_l2"_a,
      "cb_l3"_a,
      "cb_l4"_a,
      "block_size"_a = 16,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused PolarQuant dequantize: inverse_polar → inverse_precondition.

        Args:
            indices_l1..l4: quantized angle indices
            radii: block radii (float16)
            precondition: (D, D) orthogonal matrix
            cb_l1..cb_l4: codebook arrays
            block_size: polar block size

        Returns:
            Reconstructed vectors (N, D) float32
      )");
}
