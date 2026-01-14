// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#include <subsetix/mesh.hpp>
#include <subsetix/intersection/v1.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <algorithm>

namespace {

// ============================================================================
// Test helpers (CUDA-compatible)
// ============================================================================

using namespace subsetix;
// Import intersection v1 functions
using subsetix::intersection::v1::intersect_meshes;
using subsetix::intersection::v1::mesh_to;

// Helper function to create a mesh from host data
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

  host.row_keys = Mesh3DHost::RowKeyView("test_row_keys", nrows);
  host.row_ptr = Mesh3DHost::IndexView("test_row_ptr", nrows + 1);
  host.intervals = Mesh3DHost::IntervalView("test_intervals", nints);

  host.num_rows = nrows;
  host.num_intervals = nints;

  // Copy data
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

// Helper to compare host mesh with expected data
void expect_mesh_eq(
    const Mesh3DHost& actual,
    const std::vector<RowKey>& expected_keys,
    const std::vector<std::size_t>& expected_ptr,
    const std::vector<Interval>& expected_intervals) {

  EXPECT_EQ(actual.num_rows, expected_keys.size());
  EXPECT_EQ(actual.num_intervals, expected_intervals.size());

  for (std::size_t i = 0; i < actual.num_rows; ++i) {
    EXPECT_EQ(actual.row_keys(i).y, expected_keys[i].y) << "row " << i;
    EXPECT_EQ(actual.row_keys(i).z, expected_keys[i].z) << "row " << i;
    EXPECT_EQ(actual.row_ptr(i), expected_ptr[i]) << "row_ptr " << i;
  }
  EXPECT_EQ(actual.row_ptr(actual.num_rows), expected_ptr[actual.num_rows]);

  for (std::size_t i = 0; i < actual.num_intervals; ++i) {
    EXPECT_EQ(actual.intervals(i).begin, expected_intervals[i].begin) << "interval " << i;
    EXPECT_EQ(actual.intervals(i).end, expected_intervals[i].end) << "interval " << i;
  }
}

// Helper to run intersection and convert to host
Mesh3DHost run_intersection_to_host(
    const Mesh3DDevice& A,
    const Mesh3DDevice& B) {

  Mesh3DDevice result = intersect_meshes(A, B);
  return mesh_to<Kokkos::HostSpace>(result);
}

// ============================================================================
// Row-level intersection test cases (from reference implementation)
// ============================================================================

struct RowIntersectionCase {
  std::vector<Interval> intervals_a;
  std::vector<Interval> intervals_b;
  std::vector<Interval> expected;
};

// Helper to compute row intersection on device
std::vector<Interval> compute_row_intersection_device(
    const std::vector<Interval>& a,
    const std::vector<Interval>& b) {

  // Create single-row meshes
  Mesh3DDevice mesh_a = make_mesh_device({{0, 0}}, {0, a.size()}, a);
  Mesh3DDevice mesh_b = make_mesh_device({{0, 0}}, {0, b.size()}, b);

  auto result = run_intersection_to_host(mesh_a, mesh_b);

  std::vector<Interval> out;
  for (std::size_t i = 0; i < result.num_intervals; ++i) {
    out.push_back(result.intervals(i));
  }
  return out;
}

} // anonymous namespace

// ============================================================================
// Basic intersection tests
// ============================================================================

TEST(IntersectionTest, EmptyMeshes) {
  Mesh3DDevice empty_a;
  Mesh3DDevice empty_b;

  auto result = run_intersection_to_host(empty_a, empty_b);

  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

TEST(IntersectionTest, OneEmptyMesh) {
  // A has one row with one interval
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{5, 10}});

  Mesh3DDevice empty_b;

  auto result1 = run_intersection_to_host(A, empty_b);
  EXPECT_EQ(result1.num_rows, 0u);
  EXPECT_EQ(result1.num_intervals, 0u);

  auto result2 = run_intersection_to_host(empty_b, A);
  EXPECT_EQ(result2.num_rows, 0u);
  EXPECT_EQ(result2.num_intervals, 0u);
}

TEST(IntersectionTest, SingleRowIdentical) {
  // Both meshes have the same single row with the same interval
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{5, 10}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{5, 10}});

  auto result = run_intersection_to_host(A, B);

  expect_mesh_eq(result,
      {{0, 0}},           // row_keys
      {0, 1},             // row_ptr
      {{5, 10}});         // intervals
}

TEST(IntersectionTest, SingleRowPartialOverlap) {
  // A: [0, 10), B: [5, 15) -> Intersection: [5, 10)
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{0, 10}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{5, 15}});

  auto result = run_intersection_to_host(A, B);

  expect_mesh_eq(result,
      {{0, 0}},
      {0, 1},
      {{5, 10}});
}

TEST(IntersectionTest, SingleRowContained) {
  // A: [0, 20), B: [5, 10) -> Intersection: [5, 10)
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{0, 20}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{5, 10}});

  auto result = run_intersection_to_host(A, B);

  expect_mesh_eq(result,
      {{0, 0}},
      {0, 1},
      {{5, 10}});
}

TEST(IntersectionTest, SingleRowDisjoint) {
  // A: [0, 5), B: [10, 15) -> No intersection
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{0, 5}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{10, 15}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

TEST(IntersectionTest, SingleRowTouching) {
  // A: [0, 5), B: [5, 10) -> No intersection (half-open)
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{0, 5}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{5, 10}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

// ============================================================================
// Multi-row tests
// ============================================================================

TEST(IntersectionTest, MultipleRowsAllOverlap) {
  // Two rows, both overlap
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {1, 0}},
      {0, 1, 2},
      {{0, 10}, {0, 20}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}, {1, 0}},
      {0, 1, 2},
      {{5, 15}, {10, 30}});

  auto result = run_intersection_to_host(A, B);

  expect_mesh_eq(result,
      {{0, 0}, {1, 0}},
      {0, 1, 2},
      {{5, 10}, {10, 20}});
}

TEST(IntersectionTest, MultipleRowsDisjointY) {
  // A has rows y=0,1; B has rows y=2,3 -> No common rows
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {1, 0}},
      {0, 1, 2},
      {{0, 10}, {0, 10}});

  Mesh3DDevice B = make_mesh_device(
      {{2, 0}, {3, 0}},
      {0, 1, 2},
      {{0, 10}, {0, 10}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

TEST(IntersectionTest, MultipleRowsPartialOverlap) {
  // A has rows y=0,1,2; B has rows y=1,2,3
  // Common rows: y=1,2
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {1, 0}, {2, 0}},
      {0, 1, 2, 3},
      {{0, 10}, {0, 20}, {0, 30}});

  Mesh3DDevice B = make_mesh_device(
      {{1, 0}, {2, 0}, {3, 0}},
      {0, 1, 2, 3},
      {{5, 15}, {10, 25}, {0, 10}});

  auto result = run_intersection_to_host(A, B);

  expect_mesh_eq(result,
      {{1, 0}, {2, 0}},
      {0, 1, 2},
      {{5, 15}, {10, 25}});
}

TEST(IntersectionTest, MultipleIntervalsPerRow) {
  // A row has two intervals, B row has one interval covering both
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 2},
      {{0, 5}, {10, 15}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{3, 12}});

  auto result = run_intersection_to_host(A, B);

  expect_mesh_eq(result,
      {{0, 0}},
      {0, 2},
      {{3, 5}, {10, 12}});
}

TEST(IntersectionTest, MultipleIntervalsComplex) {
  // A: [0,5), [10,15), [20,25)
  // B: [3,8), [12,18), [22,28)
  // Expected: [3,5), [12,15), [22,25)
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 3},
      {{0, 5}, {10, 15}, {20, 25}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 3},
      {{3, 8}, {12, 18}, {22, 28}});

  auto result = run_intersection_to_host(A, B);

  expect_mesh_eq(result,
      {{0, 0}},
      {0, 3},
      {{3, 5}, {12, 15}, {22, 25}});
}

// ============================================================================
// 3D tests (different Z values)
// ============================================================================

TEST(IntersectionTest, DifferentZNoOverlap) {
  // Same Y, different Z
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 1},
      {{0, 10}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 1}},
      {0, 1},
      {{0, 10}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

TEST(IntersectionTest, MultipleZValues) {
  // A has (y=0, z=0) and (y=0, z=1)
  // B has (y=0, z=1) and (y=0, z=2)
  // Common: (y=0, z=1)
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {0, 1}},
      {0, 1, 2},
      {{0, 10}, {0, 20}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 1}, {0, 2}},
      {0, 1, 2},
      {{5, 15}, {0, 10}});

  auto result = run_intersection_to_host(A, B);

  expect_mesh_eq(result,
      {{0, 1}},
      {0, 1},
      {{5, 15}});
}

// ============================================================================
// Comprehensive row intersection tests
// ============================================================================

class RowIntersectionTest : public ::testing::TestWithParam<RowIntersectionCase> {};

TEST_P(RowIntersectionTest, RowIntersectionCases) {
  const auto& param = GetParam();

  std::vector<Interval> result = compute_row_intersection_device(
      param.intervals_a, param.intervals_b);

  ASSERT_EQ(result.size(), param.expected.size());

  for (std::size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i].begin, param.expected[i].begin)
        << "Mismatch at interval " << i;
    EXPECT_EQ(result[i].end, param.expected[i].end)
        << "Mismatch at interval " << i;
  }
}

// Test cases from reference implementation
INSTANTIATE_TEST_SUITE_P(
    RowIntersectionCases,
    RowIntersectionTest,
    ::testing::Values(
        // Case 0: Empty x Empty = Empty
        RowIntersectionCase{{}, {}, {}},

        // Case 1: Non-empty x Empty = Empty
        RowIntersectionCase{{{0, 2}}, {}, {}},

        // Case 2: Empty x Non-empty = Empty
        RowIntersectionCase{{}, {{0, 2}}, {}},

        // Case 3: Partial overlap
        RowIntersectionCase{{{0, 3}}, {{1, 4}}, {{1, 3}}},

        // Case 4: B contained in A
        RowIntersectionCase{{{0, 10}}, {{2, 4}, {6, 8}}, {{2, 4}, {6, 8}}},

        // Case 5: Two intervals in A, one spanning B
        RowIntersectionCase{{{0, 2}, {4, 6}}, {{1, 5}}, {{1, 2}, {4, 5}}},

        // Case 6: Disjoint
        RowIntersectionCase{{{0, 2}, {4, 6}}, {{10, 12}}, {}},

        // Case 7: Touching (half-open, no overlap)
        RowIntersectionCase{{{0, 2}}, {{2, 4}}, {}},

        // Case 8: B covers all of A
        RowIntersectionCase{{{0, 2}, {4, 6}}, {{0, 6}}, {{0, 2}, {4, 6}}},

        // Case 9: B overlaps partially with two A intervals
        RowIntersectionCase{{{0, 2}, {3, 5}, {6, 7}}, {{1, 6}}, {{1, 2}, {3, 5}}},

        // Case 10: A contained in B
        RowIntersectionCase{{{2, 4}}, {{0, 6}}, {{2, 4}}},

        // Case 11: A between B intervals
        RowIntersectionCase{{{2, 4}}, {{0, 1}, {5, 7}}, {}},

        // Case 12: Touching from left
        RowIntersectionCase{{{2, 4}}, {{0, 2}}, {}},

        // Case 13: Touching from right
        RowIntersectionCase{{{2, 4}}, {{4, 6}}, {}},

        // Case 14: Negative coordinates
        RowIntersectionCase{{{-5, -1}}, {{-3, 1}}, {{-3, -1}}}
    )
);

// ============================================================================
// PRIORITY 1: CRITICAL TESTS - Boundary Values, Properties, Edge Cases
// ============================================================================

namespace {

// Helper to check if two meshes are identical (CUDA-safe)
bool meshes_equal(const Mesh3DHost& a, const Mesh3DHost& b) {
  if (a.num_rows != b.num_rows) return false;
  if (a.num_intervals != b.num_intervals) return false;

  for (std::size_t i = 0; i < a.num_rows; ++i) {
    if (a.row_keys(i).y != b.row_keys(i).y) return false;
    if (a.row_keys(i).z != b.row_keys(i).z) return false;
    if (a.row_ptr(i) != b.row_ptr(i)) return false;
  }
  if (a.row_ptr(a.num_rows) != b.row_ptr(b.num_rows)) return false;

  for (std::size_t i = 0; i < a.num_intervals; ++i) {
    if (a.intervals(i).begin != b.intervals(i).begin) return false;
    if (a.intervals(i).end != b.intervals(i).end) return false;
  }
  return true;
}

// Helper to verify CSR invariants (CUDA-safe)
bool verify_csr_invariants(const Mesh3DHost& mesh) {
  // Check row_ptr is monotonically non-decreasing
  for (std::size_t i = 0; i < mesh.num_rows; ++i) {
    if (mesh.row_ptr(i) > mesh.row_ptr(i + 1)) {
      return false;
    }
  }

  // Check row_keys are sorted
  for (std::size_t i = 1; i < mesh.num_rows; ++i) {
    const RowKey& prev = mesh.row_keys(i - 1);
    const RowKey& curr = mesh.row_keys(i);
    if (prev.y > curr.y || (prev.y == curr.y && prev.z > curr.z)) {
      return false;
    }
  }

  // Check intervals are sorted and non-overlapping per row
  for (std::size_t row = 0; row < mesh.num_rows; ++row) {
    const std::size_t start = mesh.row_ptr(row);
    const std::size_t end = mesh.row_ptr(row + 1);
    for (std::size_t i = start; i < end; ++i) {
      if (mesh.intervals(i).begin >= mesh.intervals(i).end) {
        return false;  // Empty or inverted interval
      }
      if (i > start) {
        if (mesh.intervals(i - 1).end > mesh.intervals(i).begin) {
          return false;  // Overlapping intervals
        }
      }
    }
  }

  return true;
}

} // anonymous namespace

// Test 1: Boundary Values - Single cell at INT32_MAX
TEST(IntersectionTest, BoundaryValue_SingleCellAtMax) {
  // Single cell intervals near INT32_MAX
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 2},
      {{INT32_MAX - 2, INT32_MAX - 1}, {INT32_MAX - 1, INT32_MAX}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 2},
      {{INT32_MAX - 2, INT32_MAX - 1}, {INT32_MAX - 1, INT32_MAX}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 1u);
  EXPECT_EQ(result.num_intervals, 2u);
  EXPECT_EQ(result.intervals(0).begin, INT32_MAX - 2);
  EXPECT_EQ(result.intervals(0).end, INT32_MAX - 1);
  EXPECT_EQ(result.intervals(1).begin, INT32_MAX - 1);
  EXPECT_EQ(result.intervals(1).end, INT32_MAX);
}

// Test 2: Boundary Values - Negative coordinates
TEST(IntersectionTest, BoundaryValue_NegativeCoordinates) {
  Mesh3DDevice A = make_mesh_device(
      {{-100, -50}},
      {0, 2},
      {{-1000, -500}, {-200, -100}});

  Mesh3DDevice B = make_mesh_device(
      {{-100, -50}},
      {0, 2},
      {{-750, -400}, {-150, -50}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 1u);
  EXPECT_EQ(result.num_intervals, 2u);
  EXPECT_EQ(result.intervals(0).begin, -750);
  EXPECT_EQ(result.intervals(0).end, -500);
  EXPECT_EQ(result.intervals(1).begin, -150);
  EXPECT_EQ(result.intervals(1).end, -100);
}

// Test 3: Boundary Values - Mixed positive/negative
TEST(IntersectionTest, BoundaryValue_MixedPositiveNegative) {
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 2},
      {{-500, 0}, {0, 500}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 2},
      {{-250, 250}, {250, 750}});

  auto result = run_intersection_to_host(A, B);

  // Intersection produces 3 intervals: [-250,0), [0,250), [250,500)
  EXPECT_EQ(result.num_rows, 1u);
  EXPECT_EQ(result.num_intervals, 3u);
  EXPECT_EQ(result.intervals(0).begin, -250);
  EXPECT_EQ(result.intervals(0).end, 0);
  EXPECT_EQ(result.intervals(1).begin, 0);
  EXPECT_EQ(result.intervals(1).end, 250);
  EXPECT_EQ(result.intervals(2).begin, 250);
  EXPECT_EQ(result.intervals(2).end, 500);
}

// Test 4: Idempotency - A ∩ A = A
TEST(IntersectionTest, Idempotency_IntersectWithSelf) {
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {1, 0}, {2, 0}},
      {0, 2, 4, 6},
      {{0, 10}, {20, 30},  // Row 0
       {5, 15}, {25, 35},  // Row 1
       {10, 20}, {30, 40}}); // Row 2

  Mesh3DDevice result = intersect_meshes(A, A);
  Mesh3DHost result_host = mesh_to<Kokkos::HostSpace>(result);
  Mesh3DHost A_host = mesh_to<Kokkos::HostSpace>(A);

  EXPECT_TRUE(meshes_equal(result_host, A_host));
}

// Test 5: Commutativity - A ∩ B = B ∩ A
TEST(IntersectionTest, Commutativity_OrderDoesntMatter) {
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {1, 0}},
      {0, 2, 4},
      {{0, 15}, {25, 40},
       {10, 30}, {35, 45}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}, {1, 0}},
      {0, 2, 4},
      {{5, 20}, {30, 50},
       {5, 20}, {40, 50}});

  Mesh3DHost AB = run_intersection_to_host(A, B);
  Mesh3DHost BA = run_intersection_to_host(B, A);

  EXPECT_TRUE(meshes_equal(AB, BA));
}

// Test 6: Large-Scale Row Compaction
TEST(IntersectionTest, LargeScaleRowCompaction) {
  // Create mesh with 100 rows, only odd rows have intervals
  std::vector<RowKey> keys_a;
  std::vector<std::size_t> ptr_a;
  std::vector<Interval> iv_a;

  ptr_a.push_back(0);
  for (int i = 0; i < 100; ++i) {
    keys_a.push_back({i, 0});
    if (i % 2 == 1) {  // Only odd rows
      iv_a.push_back({i * 10, i * 10 + 5});
      ptr_a.push_back(iv_a.size());
    } else {
      ptr_a.push_back(iv_a.size());
    }
  }

  // B matches only odd rows too
  std::vector<RowKey> keys_b;
  std::vector<std::size_t> ptr_b;
  std::vector<Interval> iv_b;

  ptr_b.push_back(0);
  for (int i = 0; i < 100; ++i) {
    keys_b.push_back({i, 0});
    if (i % 2 == 1) {
      iv_b.push_back({i * 10 + 2, i * 10 + 7});
      ptr_b.push_back(iv_b.size());
    } else {
      ptr_b.push_back(iv_b.size());
    }
  }

  Mesh3DDevice A = make_mesh_device(keys_a, ptr_a, iv_a);
  Mesh3DDevice B = make_mesh_device(keys_b, ptr_b, iv_b);

  auto result = run_intersection_to_host(A, B);

  // Should have 50 rows (only odd), all with intervals
  EXPECT_EQ(result.num_rows, 50u);
  EXPECT_EQ(result.num_intervals, 50u);

  // Verify all rows have intervals
  for (std::size_t i = 0; i < result.num_rows; ++i) {
    EXPECT_LT(result.row_ptr(i), result.row_ptr(i + 1));
  }
}

// Test 7: Worst-Case Merge - Many small intervals vs one large
TEST(IntersectionTest, WorstCaseMerge_ManySmallVsOneLarge) {
  // A: 100 small intervals [0,1), [2,3), [4,5), ...
  std::vector<Interval> iv_a;
  for (int i = 0; i < 100; ++i) {
    iv_a.push_back({i * 2, i * 2 + 1});
  }

  // B: One large interval covering all
  std::vector<Interval> iv_b = {{0, 200}};

  Mesh3DDevice A = make_mesh_device({{0, 0}}, {0, 100}, iv_a);
  Mesh3DDevice B = make_mesh_device({{0, 0}}, {0, 1}, iv_b);

  auto result = run_intersection_to_host(A, B);

  // Should intersect all 100 intervals
  EXPECT_EQ(result.num_rows, 1u);
  EXPECT_EQ(result.num_intervals, 100u);

  // Verify each interval is preserved
  for (std::size_t i = 0; i < 100; ++i) {
    EXPECT_EQ(result.intervals(i).begin, i * 2);
    EXPECT_EQ(result.intervals(i).end, i * 2 + 1);
  }
}

// Test 8: Worst-Case Merge - One large vs many small
TEST(IntersectionTest, WorstCaseMerge_OneLargeVsManySmall) {
  // A: One large interval
  std::vector<Interval> iv_a = {{0, 200}};

  // B: 100 small intervals
  std::vector<Interval> iv_b;
  for (int i = 0; i < 100; ++i) {
    iv_b.push_back({i * 2, i * 2 + 1});
  }

  Mesh3DDevice A = make_mesh_device({{0, 0}}, {0, 1}, iv_a);
  Mesh3DDevice B = make_mesh_device({{0, 0}}, {0, 100}, iv_b);

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 1u);
  EXPECT_EQ(result.num_intervals, 100u);
}

// ============================================================================
// PRIORITY 2: HIGH - Single Cells, Empty Rows, Negative Y/Z, CSR Invariants
// ============================================================================

// Test 9: Single Cell Intervals
TEST(IntersectionTest, SingleCellIntervals) {
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}},
      {0, 5},
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}},
      {0, 5},
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 1u);
  EXPECT_EQ(result.num_intervals, 5u);

  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result.intervals(i).begin, i);
    EXPECT_EQ(result.intervals(i).end, i + 1);
    EXPECT_EQ(result.intervals(i).size(), 1u);
  }
}

// Test 10: Rows With No Intervals (compaction behavior)
TEST(IntersectionTest, RowsWithNoIntervals_AreCompacted) {
  // A has rows with intervals, B has same rows but no matching intervals
  std::vector<RowKey> keys = {{0, 0}, {1, 0}, {2, 0}};
  std::vector<std::size_t> ptr_a = {0, 2, 4, 6};
  std::vector<Interval> iv_a = {
    {0, 10}, {20, 30},
    {5, 15}, {25, 35},
    {10, 20}, {30, 40}
  };

  // B has same rows but disjoint intervals
  std::vector<std::size_t> ptr_b = {0, 2, 4, 6};
  std::vector<Interval> iv_b = {
    {100, 110}, {120, 130},
    {105, 115}, {125, 135},
    {110, 120}, {130, 140}
  };

  Mesh3DDevice A = make_mesh_device(keys, ptr_a, iv_a);
  Mesh3DDevice B = make_mesh_device(keys, ptr_b, iv_b);

  auto result = run_intersection_to_host(A, B);

  // All rows matched but no intervals intersect - result should be empty
  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

// Test 11: Negative Y and Z coordinates
TEST(IntersectionTest, NegativeYZ_Coordinates) {
  // Test various combinations of negative Y and Z (must be sorted!)
  // Lexicographic order: (-100,-50), (-100,50), (0,0), (100,-50)
  Mesh3DDevice A = make_mesh_device(
      {{-100, -50}, {-100, 50}, {0, 0}, {100, -50}},
      {0, 1, 2, 3, 4},
      {{0, 10}, {0, 10}, {0, 10}, {0, 10}});

  Mesh3DDevice B = make_mesh_device(
      {{-100, -50}, {-100, 50}, {0, 0}, {100, -50}},
      {0, 1, 2, 3, 4},
      {{5, 15}, {5, 15}, {5, 15}, {5, 15}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_EQ(result.num_rows, 4u);
  EXPECT_EQ(result.num_intervals, 4u);

  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(result.intervals(i).begin, 5);
    EXPECT_EQ(result.intervals(i).end, 10);
  }
}

// Test 12: Different Y Same Z (no overlap)
TEST(IntersectionTest, DifferentYSameZ_NoOverlap) {
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {1, 0}, {2, 0}},
      {0, 1, 2, 3},
      {{0, 10}, {0, 10}, {0, 10}});

  Mesh3DDevice B = make_mesh_device(
      {{10, 0}, {11, 0}, {12, 0}},
      {0, 1, 2, 3},
      {{0, 10}, {0, 10}, {0, 10}});

  auto result = run_intersection_to_host(A, B);

  // Different Y, same Z - no common rows
  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

// Test 13: Same Y Different Z (no overlap)
TEST(IntersectionTest, SameY_DifferentZ_NoOverlap) {
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {0, 1}, {0, 2}},
      {0, 1, 2, 3},
      {{0, 10}, {0, 10}, {0, 10}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 10}, {0, 11}, {0, 12}},
      {0, 1, 2, 3},
      {{0, 10}, {0, 10}, {0, 10}});

  auto result = run_intersection_to_host(A, B);

  // Same Y, different Z - no common rows
  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

// Test 14: CSR Output Invariants - Sorted and Non-Overlapping
TEST(IntersectionTest, OutputMaintainsCSRInvariants) {
  Mesh3DDevice A = make_mesh_device(
      {{0, 0}, {1, 0}, {2, 0}},
      {0, 3, 6, 9},
      {{0, 10}, {20, 30}, {40, 50},
       {5, 15}, {25, 35}, {45, 55},
       {10, 20}, {30, 40}, {50, 60}});

  Mesh3DDevice B = make_mesh_device(
      {{0, 0}, {1, 0}, {2, 0}},
      {0, 3, 6, 9},
      {{5, 15}, {25, 35}, {45, 55},
       {0, 10}, {20, 30}, {40, 50},
       {5, 15}, {35, 45}, {55, 65}});

  auto result = run_intersection_to_host(A, B);

  EXPECT_TRUE(verify_csr_invariants(result));
}

// Test 15: First Row Has No Intervals (compaction edge case)
TEST(IntersectionTest, FirstRowHasNoIntervals) {
  std::vector<RowKey> keys = {{0, 0}, {1, 0}, {2, 0}};
  std::vector<std::size_t> ptr_a = {0, 2, 4, 6};
  std::vector<Interval> iv_a = {{0, 10}, {20, 30}, {5, 15}, {25, 35}, {10, 20}, {30, 40}};

  // B intersects only rows 1 and 2
  std::vector<std::size_t> ptr_b = {0, 0, 2, 4};
  std::vector<Interval> iv_b = {{}, {5, 15}, {25, 35}, {10, 20}, {30, 40}};

  Mesh3DDevice A = make_mesh_device(keys, ptr_a, iv_a);
  Mesh3DDevice B = make_mesh_device(keys, ptr_b, iv_b);

  auto result = run_intersection_to_host(A, B);

  // Row 0 should be compacted away
  EXPECT_LE(result.num_rows, 2u);
  if (result.num_rows > 0) {
    EXPECT_TRUE(verify_csr_invariants(result));
  }
}

// Test 16: Last Row Has No Intervals (compaction edge case)
TEST(IntersectionTest, LastRowHasNoIntervals) {
  std::vector<RowKey> keys = {{0, 0}, {1, 0}, {2, 0}};
  std::vector<std::size_t> ptr_a = {0, 2, 4, 6};
  std::vector<Interval> iv_a = {{0, 10}, {20, 30}, {5, 15}, {25, 35}, {10, 20}, {30, 40}};

  // B intersects only rows 0 and 1
  std::vector<std::size_t> ptr_b = {0, 2, 4, 4};
  std::vector<Interval> iv_b = {{0, 10}, {20, 30}, {5, 15}, {25, 35}, {}};

  Mesh3DDevice A = make_mesh_device(keys, ptr_a, iv_a);
  Mesh3DDevice B = make_mesh_device(keys, ptr_b, iv_b);

  auto result = run_intersection_to_host(A, B);

  // Row 2 should be compacted away
  EXPECT_LE(result.num_rows, 2u);
  if (result.num_rows > 0) {
    EXPECT_TRUE(verify_csr_invariants(result));
  }
}

// Test 17: Alternating Rows With/Without Intervals
TEST(IntersectionTest, AlternatingRowsWithWithoutIntervals) {
  std::vector<RowKey> keys;
  std::vector<std::size_t> ptr_a;
  std::vector<Interval> iv_a;
  std::vector<std::size_t> ptr_b;
  std::vector<Interval> iv_b;

  ptr_a.push_back(0);
  ptr_b.push_back(0);

  for (int i = 0; i < 20; ++i) {
    keys.push_back({i, 0});
    if (i % 2 == 0) {
      // Even rows: A and B overlap
      iv_a.push_back({i * 10, i * 10 + 10});
      iv_b.push_back({i * 10 + 5, i * 10 + 15});
    } else {
      // Odd rows: A and B disjoint
      iv_a.push_back({i * 10, i * 10 + 10});
      iv_b.push_back({i * 10 + 100, i * 10 + 110});
    }
    ptr_a.push_back(iv_a.size());
    ptr_b.push_back(iv_b.size());
  }

  Mesh3DDevice A = make_mesh_device(keys, ptr_a, iv_a);
  Mesh3DDevice B = make_mesh_device(keys, ptr_b, iv_b);

  auto result = run_intersection_to_host(A, B);

  // Only even rows should have intervals
  EXPECT_EQ(result.num_rows, 10u);
  EXPECT_TRUE(verify_csr_invariants(result));
}
