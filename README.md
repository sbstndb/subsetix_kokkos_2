# Subsetix Kokkos

Subsetix Kokkos - Geometry processing library with Kokkos parallel backend.

[![CI](https://github.com/sbstndb/subsetix_kokkos_2/workflows/CI/badge.svg)](https://github.com/sbstndb/subsetix_kokkos_2/actions)
[![Format](https://img.shields.io/badge/code%20style-clang--format-blue.svg)](https://github.com/sbstndb/subsetix_kokkos_2)
[![Lint](https://img.shields.io/badge/static%20analysis-clang--tidy-blue.svg)](https://github.com/sbstndb/subsetix_kokkos_2)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![REUSE](https://img.shields.io/badge/reuse--check-v3.0-brightgreen.svg)](https://reuse.software/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Build Requirements](#build-requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Available Presets](#available-presets)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

Subsetix Kokkos is a modern C++20 library for geometry processing with parallel backends provided by [Kokkos](https://github.com/kokkos/kokkos). It provides efficient implementations of:

- CSR-based interval set representations
- Parallel set algebra operations
- Field operations on compressed geometries
- Multi-level AMR support

## Features

- **Parallel Execution**: Support for Serial, OpenMP, and CUDA backends
- **Modern C++**: C++20 with Kokkos for portable parallelism
- **Comprehensive Testing**: GoogleTest-based test suite
- **Performance Benchmarking**: Google Benchmark integration
- **Fast CI/CD**: Automated testing with ccache and dependency caching (2-3x faster builds)
- **Code Quality**: clang-format, clang-tidy, and pre-commit hooks

## Build Requirements

### Essential

- **CMake** >= 3.20
- **Ninja** build system
- **C++20 compiler** (g++-12 or later recommended)

### Optional (for CUDA backend)

- **CUDA toolkit** 12.x
- **NVCC** compiler
- NVIDIA GPU with Compute Capability 8.0+

### Optional (for development)

- **clang-format** - Code formatting
- **clang-tidy** - Static analysis
- **pre-commit** - Pre-commit hooks
- **ccache** - Compiler cache (faster rebuilds)

Install on Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y ninja-build ccache g++ clang-format

# For CUDA backend
sudo apt-get install -y nvidia-cuda-toolkit

# For development tools
pip install pre-commit
```

## Quick Start

### Clone and Setup

```bash
git clone git@github.com:sbstndb/subsetix_kokkos_2.git
cd subsetix_kokkos_2

# Optional: Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Configure and Build (Serial backend)

```bash
cmake --preset serial
cmake --build --preset serial
```

### Run Tests

```bash
ctest --preset serial
```

### Run Benchmarks

```bash
./build-serial/benchmarks/subsetix_benchmark_main
```

## Project Structure

```
subsetix_kokkos_2/
├── .github/
│   └── workflows/          # CI/CD workflows
├── include/subsetix/       # Public headers
│   └── config.hpp          # Configuration header
├── src/                    # Library target definition
│   └── CMakeLists.txt
├── tests/                  # GoogleTest test suites
│   ├── test_utils/         # Test utilities and helpers
│   │   └── test_helpers.hpp
│   ├── test_main.cpp       # Test entry point
│   ├── example_test.cpp    # Example tests
│   └── CMakeLists.txt
├── benchmarks/             # Google Benchmark suites
│   ├── benchmark_main.cpp  # Benchmark entry point
│   ├── example_benchmark.cpp
│   └── CMakeLists.txt
├── CMakeLists.txt          # Main CMake configuration
├── CMakePresets.json       # CMake presets for different backends
├── .clang-format           # Code formatting configuration
├── .clang-tidy             # Static analysis configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # BSD 3-Clause license
└── README.md               # This file
```

## Available Presets

### Configure Presets

The project supports a matrix of compilers (GCC, Clang) with different backends:

| Preset | Compiler | Backend | CI Status |
|--------|----------|---------|-----------|
| **Serial Backend** ||||
| `serial` | GCC (default) | Serial | ✅ Tested |
| `serial-gcc-12` | GCC 12 | Serial | ✅ Tested |
| `serial-gcc-14` | GCC 14 | Serial | ✅ Tested |
| `serial-clang` | Clang (default) | Serial | ✅ Tested |
| `serial-clang-18` | Clang 18 | Serial | ✅ Tested |
| `serial-asan` | GCC (default) | Serial + ASAN/UBSAN | ✅ Tested |
| **OpenMP Backend** ||||
| `openmp` | GCC (default) | Serial + OpenMP | ✅ Tested |
| `openmp-gcc-12` | GCC 12 | Serial + OpenMP | ✅ Tested |
| `openmp-gcc-14` | GCC 14 | Serial + OpenMP | ✅ Tested |
| `openmp-clang` | Clang (default) | Serial + OpenMP | ⚠️ Local only* |
| `openmp-clang-18` | Clang 18 | Serial + OpenMP | ⚠️ Local only* |
| **CUDA Backend** ||||
| `cuda` | GCC 12 | Serial + CUDA | ⚠️ Local only** |

*OpenMP with Clang may have CMake detection issues. To use locally, you might need:
```bash
cmake --preset openmp-clang -DOpenMP_CXX_FLAGS="-fopenmp=libomp" -DOpenMP_CXX_LIB_NAMES=omp
```

**CUDA is not tested in CI due to lack of GPU runners.

### Build Presets

Same names as configure presets (e.g., `--build --preset serial-gcc-14`)

### Test Presets

Same names as configure presets (e.g., `--test-preset openmp-gcc-12`)

### Examples

#### Configure with OpenMP

```bash
cmake --preset openmp
cmake --build --preset openmp
```

#### Configure with CUDA

```bash
cmake --preset cuda
cmake --build --preset cuda
```

#### Run a specific test

```bash
./build-serial/tests/subsetix_test_main --gtest_filter=KokkosParallelFor.SimpleRange
```

#### Run specific benchmarks

```bash
./build-serial/benchmarks/subsetix_benchmark_main --benchmark_filter=BM_ParallelFor
```

## Development

### Speed Up Builds with ccache

For faster incremental builds, use ccache to cache compilation results:

```bash
# Install ccache
sudo apt-get install ccache  # Ubuntu/Debian

# Configure CMake to use ccache
cmake --preset serial -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build --preset serial

# Check cache statistics
ccache -s
```

With ccache, subsequent builds can be **2-10x faster** for unchanged files.

### Code Formatting

Before committing, format your code:

```bash
# Format all files
find . -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

# Or use pre-commit (recommended)
pre-commit run --all-files
```

### Static Analysis

Run clang-tidy for static analysis (uses clang-18 to match CI):

```bash
cmake --preset serial-clang-18 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build --preset serial-clang-18
run-clang-tidy-18 -p build-serial-clang-18 tests/ benchmarks/
```

### Adding Tests

1. Create test files in `tests/` directory named `*_test.cpp`
2. Use GoogleTest assertions (`EXPECT_*`, `ASSERT_*`)
3. **Important**: For CUDA compatibility, do not use `KOKKOS_LAMBDA` directly inside `TEST()` macros. Use helper functions instead (see `tests/test_utils/test_helpers.hpp`).

Example:

```cpp
#include "test_utils/test_helpers.hpp"

TEST(MyTest, Example) {
  int result = subsetix::test::parallel_sum(100);
  EXPECT_EQ(result, 4950);
}
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Testing

### Run All Tests

```bash
# Serial backend
ctest --preset serial

# OpenMP backend
ctest --preset openmp

# With verbose output
ctest --preset serial --verbose
```

### Run Specific Test

```bash
./build-serial/tests/subsetix_test_main --gtest_filter=MyTest.Example
```

### Test Coverage

To generate test coverage reports (requires lcov):

```bash
cmake --preset serial -DCMAKE_CXX_FLAGS="--coverage" -DCMAKE_EXE_LINKER_FLAGS="--coverage"
cmake --build --preset serial
ctest --preset serial
lcov --capture --directory build-serial --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Development workflow
- Branching strategy (Main + Features)
- Code style and formatting
- Testing guidelines
- Pull request process

## Dependencies

All dependencies are fetched automatically via CMake FetchContent:

| Dependency | Version | License |
|------------|---------|---------|
| [Kokkos](https://github.com/kokkos/kokkos) | 5.0.1 | Apache 2.0 |
| [GoogleTest](https://github.com/google/googletest) | 1.17.0 | BSD 3-Clause |
| [Google Benchmark](https://github.com/google/benchmark) | 1.9.4 | Apache 2.0 |

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kokkos](https://github.com/kokkos/kokkos) - Performance portability layer
- [GoogleTest](https://github.com/google/googletest) - Testing framework
- [Google Benchmark](https://github.com/google/benchmark) - Benchmarking library
