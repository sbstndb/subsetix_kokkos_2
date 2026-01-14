// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#include <subsetix/mesh.hpp>
#include <subsetix/intersection/v1.hpp>

#include <benchmark/benchmark.h>

#include <vector>

namespace {

using namespace subsetix;
// Import intersection v1 functions
using subsetix::intersection::v1::intersect_meshes;
using subsetix::intersection::v1::mesh_to;

// ============================================================================
// Benchmark helpers (CUDA-compatible)
// ============================================================================

// Helper to create a mesh from host data (CUDA-safe)
Mesh3DDevice make_mesh_device(
    const std::vector<RowKey>& row_keys_vec,
    const std::vector<std::size_t>& row_ptr_vec,
    const std::vector<Interval>& intervals_vec) {

  Mesh3DHost host;
  const std::size_t nrows = row_keys_vec.size();
  const std::size_t nints = intervals_vec.size();

  if (nrows == 0) {
    return Mesh3DDevice{};
  }

  host.row_keys = Mesh3DHost::RowKeyView("bench_row_keys", nrows);
  host.row_ptr = Mesh3DHost::IndexView("bench_row_ptr", nrows + 1);
  host.intervals = Mesh3DHost::IntervalView("bench_intervals", nints);

  host.num_rows = nrows;
  host.num_intervals = nints;

  for (std::size_t i = 0; i < nrows; ++i) {
    host.row_keys(i) = row_keys_vec[i];
    host.row_ptr(i) = row_ptr_vec[i];
  }
  host.row_ptr(nrows) = row_ptr_vec[nrows];

  for (std::size_t i = 0; i < nints; ++i) {
    host.intervals(i) = intervals_vec[i];
  }

  return mesh_to<Kokkos::DefaultExecutionSpace::memory_space>(host);
}

// Helper to run intersection (CUDA-safe)
Mesh3DDevice bench_intersect(const Mesh3DDevice& A, const Mesh3DDevice& B) {
  return intersect_meshes(A, B);
}

// ============================================================================
// Benchmark: A ∩ A - 3D cube N×N×N
// ============================================================================

static void BM_Intersection_Idempotent_3DCube(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  const std::size_t n_cells = n * n * n;  // N³ cells total

  // Create mesh A with N×N rows (Y,Z combinations), 1 interval [0,N] per row (X)
  // Total: N³ cells 3D
  std::vector<RowKey> keys;
  std::vector<std::size_t> ptr;
  std::vector<Interval> iv;

  ptr.push_back(0);

  // For each (Y, Z) combination
  for (std::size_t y = 0; y < n; ++y) {
    for (std::size_t z = 0; z < n; ++z) {
      keys.push_back({static_cast<Coord>(y), static_cast<Coord>(z)});

      // Single interval [0, N) covering entire X dimension
      iv.push_back({0, static_cast<Coord>(n)});
      ptr.push_back(iv.size());
    }
  }

  Mesh3DDevice A = make_mesh_device(keys, ptr, iv);

  for (auto _ : state) {
    auto result = bench_intersect(A, A);  // A ∩ A = A
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n_cells);
}

BENCHMARK(BM_Intersection_Idempotent_3DCube)
    ->Arg(100)    // 100³ = 1,000,000 cells
    ->Arg(1000)   // 1000³ = 1,000,000,000 cells
    ->Arg(5000);  // 5000³ = 125,000,000,000 cells (~3.6 GB VRAM)

} // anonymous namespace
