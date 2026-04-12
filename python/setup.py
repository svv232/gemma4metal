"""TurboQuant Python extension using MLX's extension framework."""
from mlx.extension import CMakeExtension, CMakeBuild
from setuptools import setup

if __name__ == "__main__":
    setup(
        name="turboquant",
        version="0.1.0",
        description="Fused int4 SDPA Metal kernel for MLX",
        ext_modules=[CMakeExtension("turboquant_ext")],
        cmdclass={"build_ext": CMakeBuild},
        python_requires=">=3.8",
    )
