cmake_minimum_required(VERSION 3.27)
project(turboquant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# --- MLX dependency ---
set(MLX_SOURCE_DIR "/tmp/mlx-source")
set(MLX_BUILD_DIR "/tmp/mlx-source/build")

add_library(mlx STATIC IMPORTED)
set_target_properties(mlx PROPERTIES
  IMPORTED_LOCATION "${MLX_BUILD_DIR}/libmlx.a"
  INTERFACE_INCLUDE_DIRECTORIES "${MLX_SOURCE_DIR}"
)

find_library(METAL_LIB Metal)
find_library(METALKIT_LIB MetalKit)
find_library(FOUNDATION_LIB Foundation)
find_library(ACCELERATE_LIB Accelerate)

# --- TurboQuant library ---
add_library(turboquant_lib lib/turboquant.cpp)
target_include_directories(turboquant_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/lib
  ${MLX_SOURCE_DIR}
  ${MLX_BUILD_DIR}/_deps/metal_cpp-src
)
target_link_libraries(turboquant_lib PUBLIC
  mlx ${METAL_LIB} ${METALKIT_LIB} ${FOUNDATION_LIB} ${ACCELERATE_LIB}
)

# --- Metal kernels ---
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib
  COMMAND xcrun -sdk macosx metal -c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/turboquant.metal
    -o ${CMAKE_CURRENT_BINARY_DIR}/turboquant.air
  COMMAND xcrun -sdk macosx metallib
    ${CMAKE_CURRENT_BINARY_DIR}/turboquant.air
    -o ${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lib/turboquant.metal
  COMMENT "Compiling Metal kernels (sdpa_int4 + PolarQuant)"
)
add_custom_target(metallib DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib)

# --- Helper macro ---
macro(add_tq_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE turboquant_lib)
  add_dependencies(${name} metallib)
  target_compile_definitions(${name} PRIVATE
    METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/turboquant.metallib"
  )
endmacro()

# --- Targets ---
add_tq_executable(gemma4 engine/gemma4_multilayer.cpp)
add_tq_executable(test_sdpa_int4 tests/test_sdpa_int4.cpp)
