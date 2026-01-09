cmake_minimum_required(VERSION 3.20)
project(subsetix_kokkos LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Options
# ============================================================================

option(SUBSETIX_KOKKOS_SERIAL "Enable Kokkos Serial backend" ON)
option(SUBSETIX_KOKKOS_OPENMP "Enable Kokkos OpenMP backend" OFF)
option(SUBSETIX_KOKKOS_CUDA "Enable Kokkos CUDA backend" OFF)

option(SUBSETIX_BUILD_TESTS "Build tests" ON)
option(SUBSETIX_BUILD_BENCHMARKS "Build benchmarks" ON)

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

# ============================================================================
# Sanitizers
# ============================================================================

if(SUBSETIX_ENABLE_SANITIZERS)
  message(STATUS "Enabling address and undefined behavior sanitizers")
  add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address -fsanitize=undefined)
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
