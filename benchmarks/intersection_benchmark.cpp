// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#include <subsetix/mesh.hpp>
#include <subsetix/intersection.hpp>

#include <benchmark/benchmark.h>

#include <vector>

namespace {

using namespace subsetix;

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
// Benchmark: A ∩ A (idempotent) - Simple scaling test
// ============================================================================

static void BM_Intersection_Idempotent_Scaling(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  // Create mesh A with n rows, 1 interval per row
  std::vector<RowKey> keys(n);
  std::vector<std::size_t> ptr(n + 1);
  std::vector<Interval> iv;

  ptr[0] = 0;
  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = {static_cast<Coord>(i), 0};
    iv.push_back({static_cast<Coord>(i), static_cast<Coord>(i + 1)});
    ptr[i + 1] = iv.size();
  }

  Mesh3DDevice A = make_mesh_device(keys, ptr, iv);

  for (auto _ : state) {
    auto result = bench_intersect(A, A);  // A ∩ A = A
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Intersection_Idempotent_Scaling)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000);

} // anonymous namespace
