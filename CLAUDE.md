# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Subsetix Kokkos is a C++20 geometry processing library with portable parallel backends via Kokkos. The project implements CSR-based interval set representations, parallel set algebra operations, field operations on compressed geometries, and multi-level AMR support.

**Key characteristics:**
- Header-only library (INTERFACE library in CMake)
- C++20 with Kokkos 5.0.1 for parallel backends
- Supports Serial, OpenMP, and CUDA execution spaces
- All dependencies fetched automatically via FetchContent

## Build and Development Commands

### Quick Start

```bash
# Configure and build (Serial backend - default)
cmake --preset serial
cmake --build --preset serial

# Run tests
ctest --preset serial

# Run benchmarks
./build-serial/benchmarks/subsetix_benchmark_main
```

### Available Presets

The project supports a matrix of compilers and backends via CMake presets:

**Serial Backend:** `serial`, `serial-gcc-12`, `serial-gcc-14`, `serial-clang`, `serial-clang-18`, `serial-asan`

**OpenMP Backend:** `openmp`, `openmp-gcc-12`, `openmp-gcc-14`, `openmp-clang`, `openmp-clang-18`

**CUDA Backend:** `cuda` (local only - not tested in CI)

Note: OpenMP with Clang may require `-DOpenMP_CXX_FLAGS="-fopenmp=libomp" -DOpenMP_CXX_LIB_NAMES=omp`

### Testing

```bash
# Run all tests
ctest --preset serial

# Run with verbose output
ctest --preset serial --verbose

# Run specific test
./build-serial/tests/subsetix_test_main --gtest_filter=MyTest.Example

# Run specific test suite
./build-serial/tests/subsetix_test_main --gtest_filter=MyTest.*
```

### Benchmarking

```bash
# Run all benchmarks
./build-openmp/benchmarks/subsetix_benchmark_main

# Run specific benchmark
./build-openmp/benchmarks/subsetix_benchmark_main --benchmark_filter=BM_ParallelFor

# Run with custom minimum time
./build-openmp/benchmarks/subsetix_benchmark_main --benchmark_min_time=0.1
```

### Code Quality

```bash
# Format all files (uses project .clang-format)
find . -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

# Run pre-commit hooks (recommended)
pre-commit run --all-files

# Static analysis with clang-tidy
cmake --preset serial -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
run-clang-tidy.py -p build-serial tests/ benchmarks/
```

### Performance Optimization

Use ccache for 2-10x faster incremental builds:

```bash
sudo apt-get install ccache
cmake --preset serial -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build --preset serial
ccache -s  # Check cache statistics
```

## Architecture

### Project Structure

```
include/subsetix/         # Public headers (header-only library)
src/CMakeLists.txt        # Core library INTERFACE target
tests/                    # GoogleTest suites
tests/test_utils/         # Test utilities and helpers
benchmarks/               # Google Benchmark suites
CMakeLists.txt            # Main CMake configuration
CMakePresets.json         # Build presets for backends/compilers
```

### Library Design

The project uses a **header-only library pattern**:
- `subsetix::core` is an INTERFACE library
- Headers exposed from `include/subsetix/`
- Links against `Kokkos::kokkos`
- Installation rules for packaging

### Critical CUDA Testing Pattern

**This is the most important pattern to understand:**

`KOKKOS_LAMBDA` cannot be used directly inside `TEST()` macros due to CUDA compilation requirements. Always extract kernel logic into helper functions.

**Incorrect:**
```cpp
TEST(MyTest, BadExample) {
  Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(int i, int& sum) {
    sum += i;
  }, result);
}
```

**Correct:**
```cpp
// Helper function defined outside TEST() macro
int compute_sum(int n) {
  int result = 0;
  Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(int i, int& sum) {
    sum += i;
  }, result);
  return result;
}

TEST(MyTest, GoodExample) {
  EXPECT_EQ(compute_sum(100), 4950);
}
```

Helper functions should be placed in `tests/test_utils/test_helpers.hpp` for reuse.

## Code Style

- **Indentation:** 2 spaces, no tabs
- **Line width:** 100 characters
- **Naming:**
  - Types: `CamelCase`
  - Functions/variables: `camelBack`
  - Constants: `UPPER_CASE`
  - Private members: `camelBack` (no prefix)
- **Formatting:** Configuration in `.clang-format` (Google style with customizations)
- **Include sorting:** Regrouped by category (std, Kokkos, GoogleTest, project, other)

## Commit Convention

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`

## Dependencies

All fetched automatically via FetchContent:
- **Kokkos 5.0.1** - Parallel backends (Apache 2.0)
- **GoogleTest 1.17.0** - Testing (BSD 3-Clause)
- **Google Benchmark 1.9.4** - Benchmarking (Apache 2.0)

## CI Matrix

The GitHub Actions CI tests:
- Serial: GCC 12, GCC 14, Clang 18
- OpenMP: GCC 12, GCC 14
- Sanitizers: GCC 14 with ASAN/UBSAN
- Benchmarks: GCC 14 + OpenMP
- Code style: clang-format verification

CUDA is not tested in CI due to lack of GPU runners.
