cmake_minimum_required(VERSION 3.27)
project(turboquant_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SHARED_LIBS "Build as shared library" ON)

# Dependencies
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Pin nanobind to same version as MLX 0.31.1 (ABI must match exactly)
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.10.2
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nanobind)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# TurboQuant library
add_library(turboquant_lib)
target_sources(turboquant_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../lib/turboquant.cpp)
target_include_directories(turboquant_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../lib)
target_link_libraries(turboquant_lib PUBLIC mlx)

# Metal kernels
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET turboquant_metallib
    TITLE turboquant
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/../lib/turboquant.metal
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../lib ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(turboquant_lib turboquant_metallib)
endif()

# Python bindings
nanobind_add_module(
  turboquant_ext
  NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/tq_bindings.cpp)
target_link_libraries(turboquant_ext PRIVATE turboquant_lib)

if(BUILD_SHARED_LIBS)
  target_link_options(turboquant_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
