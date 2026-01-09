# Subsetix Kokkos

Subsetix Kokkos - Geometry processing library with Kokkos parallel backend.

## Build Requirements

- CMake >= 3.20
- C++20 compiler (g++-12 or later recommended)
- Ninja build system
- CUDA toolkit (optional, for CUDA backend)

## Quick Start

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

## Available Presets

### Configure Presets

- `serial` - Serial backend only (default)
- `serial-asan` - Serial with sanitizers (for debugging)
- `openmp` - OpenMP backend
- `cuda` - CUDA backend

### Build Presets

Same names as configure presets: `serial`, `serial-asan`, `openmp`, `cuda`

### Test Presets

Same names as configure presets: `serial`, `serial-asan`, `openmp`, `cuda`

## Project Structure

```
subsetix_kokkos_2/
├── CMakeLists.txt          # Main CMake configuration
├── CMakePresets.json       # CMake presets for different backends
├── include/subsetix/       # Public headers
├── src/                    # Library target definition
├── tests/                  # GoogleTest test suites
└── benchmarks/             # Google Benchmark micro-benchmarks
```

## Dependencies

All dependencies are fetched automatically via CMake FetchContent:

- **Kokkos 5.0.1** - Parallel programming framework
- **GoogleTest 1.15.0** - Testing framework (when building tests)
- **Google Benchmark 1.8.4** - Benchmarking framework (when building benchmarks)

## Examples

### Configure with OpenMP

```bash
cmake --preset openmp
cmake --build --preset openmp
```

### Configure with CUDA

```bash
cmake --preset cuda
cmake --build --preset cuda
```

### Run a specific test

```bash
./build-serial/tests/subsetix_test_main --gtest_filter=KokkosInitialization.BasicTest
```

### Run specific benchmarks

```bash
./build-serial/benchmarks/subsetix_benchmark_main --benchmark_filter=BM_ParallelFor
```
