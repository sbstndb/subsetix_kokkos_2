// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#include <subsetix/mesh.hpp>
#include <subsetix/intersection.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <algorithm>

namespace {

// ============================================================================
// Test helpers (CUDA-compatible)
// ============================================================================

using namespace subsetix;

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

  IntersectionWorkspace ws;
  Mesh3DDevice result = intersect_meshes(A, B, ws);
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
