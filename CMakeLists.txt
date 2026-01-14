cmake_minimum_required(VERSION 3.20)
project(subsetix_kokkos
  VERSION 0.1.0
  DESCRIPTION "Geometry processing library with Kokkos parallel backends"
  HOMEPAGE_URL "https://github.com/sbstndb/subsetix_kokkos_2"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Options
# ============================================================================

option(SUBSETIX_KOKKOS_SERIAL "Enable Kokkos Serial backend" ON)
option(SUBSETIX_KOKKOS_OPENMP "Enable Kokkos OpenMP backend" OFF)
option(SUBSETIX_KOKKOS_CUDA "Enable Kokkos CUDA backend" OFF)

# Validate that at least one backend is enabled
if(NOT SUBSETIX_KOKKOS_SERIAL AND NOT SUBSETIX_KOKKOS_OPENMP AND NOT SUBSETIX_KOKKOS_CUDA)
  message(FATAL_ERROR "At least one Kokkos backend must be enabled (SERIAL, OPENMP, or CUDA)")
endif()

option(SUBSETIX_BUILD_TESTS "Build tests" ON)
option(SUBSETIX_BUILD_BENCHMARKS "Build benchmarks" ON)

# ============================================================================
# Intersection Implementation Selection
# ============================================================================

option(SUBSETIX_ENABLE_INTERSECTION_V1 "Enable intersection v1" ON)
option(SUBSETIX_ENABLE_INTERSECTION_V2 "Enable intersection v2 (future)" OFF)
option(SUBSETIX_ENABLE_INTERSECTION_CUDA_OPT "Enable CUDA optimized intersection (future)" OFF)

# Optional dependencies for alternative implementations
option(SUBSETIX_USE_THRUST "Enable Thrust library for alternative implementations" OFF)
option(SUBSETIX_USE_CUB "Enable CUB library for CUDA optimizations" OFF)

# Validate that at least one intersection implementation is enabled
if(NOT SUBSETIX_ENABLE_INTERSECTION_V1 AND NOT SUBSETIX_ENABLE_INTERSECTION_V2 AND NOT SUBSETIX_ENABLE_INTERSECTION_CUDA_OPT)
  message(FATAL_ERROR "At least one intersection implementation must be enabled (V1, V2, or CUDA_OPT)")
endif()

option(SUBSETIX_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

# ============================================================================
# Dependencies via FetchContent
# ============================================================================

include(FetchContent)

# ----------------------------------------------------------------------------
# Kokkos
# ----------------------------------------------------------------------------

message(STATUS "Fetching Kokkos 5.0.1...")

FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 5.0.1
  GIT_SHALLOW TRUE
)

# Set Kokkos options
set(Kokkos_ENABLE_SERIAL ${SUBSETIX_KOKKOS_SERIAL} CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP ${SUBSETIX_KOKKOS_OPENMP} CACHE BOOL "")
set(Kokkos_ENABLE_CUDA ${SUBSETIX_KOKKOS_CUDA} CACHE BOOL "")

# Disable Kokkos tests and examples
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "")
set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "")

FetchContent_MakeAvailable(kokkos)

# ----------------------------------------------------------------------------
# GoogleTest (for testing)
# ----------------------------------------------------------------------------

if(SUBSETIX_BUILD_TESTS)
  message(STATUS "Fetching GoogleTest 1.17.0...")

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
  )

  # Disable gtest own tests
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(googletest)
endif()

# ----------------------------------------------------------------------------
# Google Benchmark (for benchmarking)
# ----------------------------------------------------------------------------

if(SUBSETIX_BUILD_BENCHMARKS)
  message(STATUS "Fetching Google Benchmark 1.9.4...")

  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
    GIT_SHALLOW TRUE
  )

  # Disable benchmark own tests
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "")
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")

  FetchContent_MakeAvailable(benchmark)
endif()

# ----------------------------------------------------------------------------
# Optional dependencies for alternative implementations
# ----------------------------------------------------------------------------

# Thrust (for alternative implementations)
if(SUBSETIX_USE_THRUST)
  message(STATUS "Fetching Thrust (CUB) for alternative implementations...")

  # Thrust is now included as part of CUDA toolkit, but we can also fetch it
  # For non-CUDA builds, we fetch thrust separately
  if(NOT SUBSETIX_KOKKOS_CUDA)
    FetchContent_Declare(
      thrust
      GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
      GIT_TAG cuda-12.2.1
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(thrust)
  else()
    message(STATUS "  Using Thrust from CUDA toolkit")
  endif()
endif()

# CUB (for CUDA optimizations)
if(SUBSETIX_USE_CUB)
  message(STATUS "Fetching CUB for CUDA optimizations...")

  FetchContent_Declare(
    cub
    GIT_REPOSITORY https://github.com/NVIDIA/cub.git
    GIT_TAG 2.1.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(cub)
endif()

# ============================================================================
# Sanitizers
# ============================================================================

# Sanitizer flags for targets (tests, benchmarks)
# Note: NOT applied globally because this is a header-only library
if(SUBSETIX_ENABLE_SANITIZERS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Enabling address and undefined behavior sanitizers")
    set(SUBSETIX_SANITIZER_COMPILE_FLAGS "-fsanitize=address" "-fsanitize=undefined" "-fno-omit-frame-pointer")
    set(SUBSETIX_SANITIZER_LINK_FLAGS "-fsanitize=address" "-fsanitize=undefined")
  else()
    message(WARNING "Sanitizers not supported for ${CMAKE_CXX_COMPILER_ID} compiler")
  endif()
endif()

# ============================================================================
# Subdirectories
# ============================================================================

add_subdirectory(src)

if(SUBSETIX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SUBSETIX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ============================================================================
# Print configuration summary
# ============================================================================

message(STATUS "")
message(STATUS "=== Subsetix Kokkos Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Backends:")
message(STATUS "  Serial:  ${SUBSETIX_KOKKOS_SERIAL}")
message(STATUS "  OpenMP:  ${SUBSETIX_KOKKOS_OPENMP}")
message(STATUS "  CUDA:    ${SUBSETIX_KOKKOS_CUDA}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  Tests:       ${SUBSETIX_BUILD_TESTS}")
message(STATUS "  Benchmarks:  ${SUBSETIX_BUILD_BENCHMARKS}")
message(STATUS "  Sanitizers:  ${SUBSETIX_ENABLE_SANITIZERS}")
message(STATUS "=====================================")
message(STATUS "")
