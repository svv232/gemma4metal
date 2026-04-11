cmake_minimum_required(VERSION 3.27)
project(turboquant_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SHARED_LIBS "Build as shared library" ON)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mlx; print(mlx.__path__[0])"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
set(MLX_DIR "${MLX_ROOT}/share/cmake/MLX")
find_package(MLX CONFIG REQUIRED)

# C++ library
add_library(turboquant_lib)
target_sources(turboquant_lib PUBLIC turboquant/turboquant.cpp)
target_include_directories(turboquant_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(turboquant_lib PUBLIC mlx)

# Metal kernels
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET turboquant_metallib
    TITLE turboquant
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/turboquant/turboquant.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(turboquant_lib turboquant_metallib)
endif()

# Python bindings — NB_DOMAIN mlx is the key!
nanobind_add_module(
  turboquant_ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  turboquant/bindings.cpp
)
target_link_libraries(turboquant_ext PRIVATE turboquant_lib)

if(BUILD_SHARED_LIBS)
  target_link_options(turboquant_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
