"""Setup for TurboQuant C++ MLX extension."""

from mlx.extension import CMakeExtension, CMakeBuild
from setuptools import setup

if __name__ == "__main__":
    setup(
        name="turboquant_ext",
        version="0.1.0",
        description="TurboQuant: Fused PolarQuant KV cache compression for MLX",
        ext_modules=[CMakeExtension("turboquant_ext")],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
        python_requires=">=3.9",
    )
