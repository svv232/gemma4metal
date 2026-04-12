cmake_minimum_required(VERSION 3.27)
project(turboquant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Use source-built MLX
set(MLX_SOURCE_DIR "/tmp/mlx-source")
set(MLX_BUILD_DIR "/tmp/mlx-source/build")

# MLX library and headers
add_library(mlx STATIC IMPORTED)
set_target_properties(mlx PROPERTIES
  IMPORTED_LOCATION "${MLX_BUILD_DIR}/libmlx.a"
  INTERFACE_INCLUDE_DIRECTORIES "${MLX_SOURCE_DIR}"
)

# Find Metal framework
find_library(METAL_LIB Metal)
find_library(METALKIT_LIB MetalKit)
find_library(FOUNDATION_LIB Foundation)
find_library(ACCELERATE_LIB Accelerate)

# TurboQuant library
add_library(turboquant_lib turboquant/turboquant.cpp)
target_include_directories(turboquant_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${MLX_SOURCE_DIR}
  ${MLX_BUILD_DIR}/_deps/metal_cpp-src
)
target_link_libraries(turboquant_lib PUBLIC
  mlx
  ${METAL_LIB}
  ${METALKIT_LIB}
  ${FOUNDATION_LIB}
  ${ACCELERATE_LIB}
)

# Compile Metal kernels to metallib
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib
  COMMAND xcrun -sdk macosx metal -c
    ${CMAKE_CURRENT_SOURCE_DIR}/turboquant/turboquant.metal
    -I ${MLX_SOURCE_DIR}
    -o ${CMAKE_CURRENT_BINARY_DIR}/turboquant.air
  COMMAND xcrun -sdk macosx metallib
    ${CMAKE_CURRENT_BINARY_DIR}/turboquant.air
    -o ${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/turboquant/turboquant.metal
  COMMENT "Compiling TurboQuant Metal kernels"
)
add_custom_target(turboquant_metallib DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib)

# Test executable — pure C++, no Python!
add_executable(turboquant_test turboquant/test_main.cpp)
target_link_libraries(turboquant_test PRIVATE turboquant_lib)
add_dependencies(turboquant_test turboquant_metallib)
target_compile_definitions(turboquant_test PRIVATE
  METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib"
)

# Gemma 4 test
add_executable(gemma4_test turboquant/gemma4_test.cpp)
target_link_libraries(gemma4_test PRIVATE turboquant_lib)
add_dependencies(gemma4_test turboquant_metallib)
target_compile_definitions(gemma4_test PRIVATE
  METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib"
)

# Gemma 4 full inference
add_executable(gemma4_inference turboquant/gemma4_inference.cpp)
target_link_libraries(gemma4_inference PRIVATE turboquant_lib)
add_dependencies(gemma4_inference turboquant_metallib)
target_compile_definitions(gemma4_inference PRIVATE
  METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib"
)

# Gemma 4 block test
add_executable(gemma4_block turboquant/gemma4_block.cpp)
target_link_libraries(gemma4_block PRIVATE turboquant_lib)
add_dependencies(gemma4_block turboquant_metallib)
target_compile_definitions(gemma4_block PRIVATE
  METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib"
)
