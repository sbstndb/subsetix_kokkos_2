[![CI](https://github.com/sbstndb/subsetix_kokkos_2/workflows/CI/badge.svg)](https://github.com/sbstndb/subsetix_kokkos_2/actions)
[![codecov](https://codecov.io/gh/sbstndb/subsetix_kokkos_2/branch/main/graph/badge.svg)](https://codecov.io/gh/sbstndb/subsetix_kokkos_2)
[![Format](https://img.shields.io/badge/code%20style-clang--format-blue.svg)](https://github.com/sbstndb/subsetix_kokkos_2)
[![Lint](https://img.shields.io/badge/static%20analysis-clang--tidy-blue.svg)](https://github.com/sbstndb/subsetix_kokkos_2)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![REUSE](https://img.shields.io/badge/reuse--check-v3.0-brightgreen.svg)](https://reuse.software/)
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)](https://github.com/sbstndb/subsetix_kokkos_2)

# Subsetix Kokkos

C++20 geometry processing library with Kokkos parallel backends (Serial/OpenMP/CUDA).

## Quick Start

```bash
git clone git@github.com:sbstndb/subsetix_kokkos_2.git
cd subsetix_kokkos_2
cmake --preset serial && cmake --build --preset serial
ctest --preset serial
```

## Presets

**Serial:** `serial`, `serial-gcc-12/14`, `serial-clang-18`, `serial-asan`

**OpenMP:** `openmp`, `openmp-gcc-12/14`

**CUDA:** `cuda` (local only)

## Dev

```bash
# Format + lint
pre-commit run --all-files

# clang-tidy
cmake --preset serial-clang-18 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
run-clang-tidy-18 -p build-serial-clang-18 tests/ benchmarks/

# ccache (2-10x faster builds)
cmake --preset serial -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

## Important

Tests using `KOKKOS_LAMBDA` must extract logic into helper functions (CUDA compatibility). See `tests/test_utils/test_helpers.hpp`.
