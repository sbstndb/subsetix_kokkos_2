// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#include <subsetix/mesh.hpp>
#include <subsetix/intersection.hpp>

#include <benchmark/benchmark.h>

#include <vector>
#include <random>
#include <algorithm>

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

// Generate a mesh with random intervals (host-side)
struct RandomMeshParams {
  std::size_t num_rows = 100;
  std::size_t intervals_per_row = 5;
  Coord y_start = 0;
  Coord z_start = 0;
  Coord interval_length = 10;
  Coord interval_spacing = 20;
  double overlap_probability = 0.5;  // For generating overlapping meshes
};

Mesh3DDevice generate_random_mesh(const RandomMeshParams& params, std::uint64_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> overlap_dist(0.0, 1.0);

  std::vector<RowKey> row_keys;
  std::vector<std::size_t> row_ptr;
  std::vector<Interval> intervals;

  row_ptr.push_back(0);

  for (std::size_t i = 0; i < params.num_rows; ++i) {
    row_keys.push_back({static_cast<Coord>(params.y_start + i),
                        static_cast<Coord>(params.z_start)});

    std::size_t row_start = intervals.size();

    for (std::size_t j = 0; j < params.intervals_per_row; ++j) {
      Coord x_start = static_cast<Coord>(j * params.interval_spacing);
      intervals.push_back({x_start, x_start + params.interval_length});
    }

    row_ptr.push_back(intervals.size());
  }

  return make_mesh_device(row_keys, row_ptr, intervals);
}

// Generate two meshes with controlled overlap
struct OverlappingMeshPair {
  Mesh3DDevice A;
  Mesh3DDevice B;
};

OverlappingMeshPair generate_overlapping_meshes(
    std::size_t num_rows,
    std::size_t intervals_per_row,
    double overlap_ratio,  // 0.0 = disjoint, 1.0 = identical
    std::uint64_t seed) {

  std::mt19937 gen_a(seed);
  std::mt19937 gen_b(seed + 1);

  std::vector<RowKey> row_keys_a, row_keys_b;
  std::vector<std::size_t> row_ptr_a, row_ptr_b;
  std::vector<Interval> intervals_a, intervals_b;

  row_ptr_a.push_back(0);
  row_ptr_b.push_back(0);

  const Coord interval_length = 10;
  const Coord interval_spacing = 20;

  for (std::size_t i = 0; i < num_rows; ++i) {
    row_keys_a.push_back({static_cast<Coord>(i), 0});
    row_keys_b.push_back({static_cast<Coord>(i), 0});

    std::size_t row_start_a = intervals_a.size();
    std::size_t row_start_b = intervals_b.size();

    for (std::size_t j = 0; j < intervals_per_row; ++j) {
      Coord x_base = static_cast<Coord>(j * interval_spacing);

      // A starts at 0
      intervals_a.push_back({x_base, x_base + interval_length});

      // B is shifted by overlap_ratio * interval_spacing
      Coord shift = static_cast<Coord>(overlap_ratio * interval_spacing);
      intervals_b.push_back({x_base + shift, x_base + shift + interval_length});
    }

    row_ptr_a.push_back(intervals_a.size());
    row_ptr_b.push_back(intervals_b.size());
  }

  return {make_mesh_device(row_keys_a, row_ptr_a, intervals_a),
          make_mesh_device(row_keys_b, row_ptr_b, intervals_b)};
}

// Helper to run intersection (CUDA-safe)
Mesh3DDevice bench_intersect(const Mesh3DDevice& A, const Mesh3DDevice& B) {
  return intersect_meshes(A, B);
}

// ============================================================================
// Benchmarks: Single row, varying intervals
// ============================================================================

static void BM_Intersection_SingleRow_FewIntervals(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  auto A = make_mesh_device({{0, 0}}, {0, n}, std::vector<Interval>(n, {0, 10}));
  auto B = make_mesh_device({{0, 0}}, {0, n}, std::vector<Interval>(n, {5, 15}));

  for (auto _ : state) {
    auto result = bench_intersect(A, B);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Intersection_SingleRow_FewIntervals)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);

static void BM_Intersection_SingleRow_ManyIntervals(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  std::vector<Interval> iv_a(n), iv_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    iv_a[i] = {static_cast<Coord>(i * 20), static_cast<Coord>(i * 20 + 10)};
    iv_b[i] = {static_cast<Coord>(i * 20 + 5), static_cast<Coord>(i * 20 + 15)};
  }

  auto A = make_mesh_device({{0, 0}}, {0, n}, iv_a);
  auto B = make_mesh_device({{0, 0}}, {0, n}, iv_b);

  for (auto _ : state) {
    auto result = bench_intersect(A, B);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Intersection_SingleRow_ManyIntervals)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);

// ============================================================================
// Benchmarks: Multiple rows
// ============================================================================

static void BM_Intersection_MultipleRows(benchmark::State& state) {
  const std::size_t n_rows = static_cast<std::size_t>(state.range(0));
  const std::size_t n_ints = 4;

  std::vector<RowKey> keys(n_rows);
  std::vector<std::size_t> ptr(n_rows + 1);
  std::vector<Interval> iv_a, iv_b;

  ptr[0] = 0;
  for (std::size_t i = 0; i < n_rows; ++i) {
    keys[i] = {static_cast<Coord>(i), 0};
    for (std::size_t j = 0; j < n_ints; ++j) {
      Coord x = static_cast<Coord>(j * 20);
      iv_a.push_back({x, x + 10});
      iv_b.push_back({x + 5, x + 15});
    }
    ptr[i + 1] = iv_a.size();
  }

  auto A = make_mesh_device(keys, ptr, iv_a);
  auto B = make_mesh_device(keys, ptr, iv_b);

  for (auto _ : state) {
    auto result = bench_intersect(A, B);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n_rows * n_ints);
}

BENCHMARK(BM_Intersection_MultipleRows)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

// ============================================================================
// Benchmarks: Overlap scenarios
// ============================================================================

static void BM_Intersection_OverlapScenarios(benchmark::State& state) {
  const double overlap = static_cast<double>(state.range(0)) / 100.0;
  const std::size_t n_rows = 100;
  const std::size_t n_ints = 10;

  auto [A, B] = generate_overlapping_meshes(n_rows, n_ints, overlap, 42);

  for (auto _ : state) {
    auto result = bench_intersect(A, B);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n_rows * n_ints);
}

BENCHMARK(BM_Intersection_OverlapScenarios)
    ->Arg(0)    // Disjoint
    ->Arg(25)   // Slight overlap
    ->Arg(50)   // 50% overlap
    ->Arg(75)   // High overlap
    ->Arg(100); // Identical

// ============================================================================
// Benchmarks: Different mesh sizes
// ============================================================================

static void BM_Intersection_SizeScaling(benchmark::State& state) {
  const std::size_t n_rows = static_cast<std::size_t>(state.range(0));
  const std::size_t n_ints = static_cast<std::size_t>(state.range(1));

  std::vector<RowKey> keys(n_rows);
  std::vector<std::size_t> ptr(n_rows + 1);
  std::vector<Interval> iv;

  ptr[0] = 0;
  for (std::size_t i = 0; i < n_rows; ++i) {
    keys[i] = {static_cast<Coord>(i), 0};
    for (std::size_t j = 0; j < n_ints; ++j) {
      Coord x = static_cast<Coord>(j * 20);
      iv.push_back({x, x + 10});
    }
    ptr[i + 1] = iv.size();
  }

  auto A = make_mesh_device(keys, ptr, iv);
  auto B = make_mesh_device(keys, ptr, iv);

  for (auto _ : state) {
    auto result = bench_intersect(A, B);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n_rows * n_ints);
}

BENCHMARK(BM_Intersection_SizeScaling)
    ->Args({10, 10})
    ->Args({100, 10})
    ->Args({1000, 10})
    ->Args({10000, 10})
    ->Args({10, 100})
    ->Args({100, 100})
    ->Args({1000, 100});

// ============================================================================
// Benchmarks: Empty results
// ============================================================================

static void BM_Intersection_DisjointRows(benchmark::State& state) {
  const std::size_t n_rows = static_cast<std::size_t>(state.range(0));

  std::vector<RowKey> keys_a(n_rows), keys_b(n_rows);
  std::vector<std::size_t> ptr(n_rows + 1, 0);
  std::vector<Interval> iv_a, iv_b;

  for (std::size_t i = 0; i < n_rows; ++i) {
    keys_a[i] = {static_cast<Coord>(i), 0};
    keys_b[i] = {static_cast<Coord>(i + n_rows), 0};  // Offset rows
    iv_a.push_back({0, 10});
    iv_b.push_back({0, 10});
    ptr[i + 1] = static_cast<std::size_t>(i + 1);
  }

  auto A = make_mesh_device(keys_a, ptr, iv_a);
  auto B = make_mesh_device(keys_b, ptr, iv_b);

  for (auto _ : state) {
    auto result = bench_intersect(A, B);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n_rows);
}

BENCHMARK(BM_Intersection_DisjointRows)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

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
