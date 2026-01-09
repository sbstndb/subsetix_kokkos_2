// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace subsetix::test {

// ============================================================================
// CUDA-Compatible Test Helpers
// ============================================================================

// IMPORTANT: When writing tests with Kokkos parallel constructs, you CANNOT
// use KOKKOS_LAMBDA directly inside TEST() macros. This is because TEST()
// creates a class with TestBody() as a private member, and CUDA doesn't allow
// __host__ __device__ lambdas in private member functions.
//
// Instead, use one of these approaches:
// 1. Use the helper functions below
// 2. Extract your kernel logic into a separate function (outside the TEST())
// 3. Use KOKKOS_LAMBDA only in free functions, not in TEST() directly

// ============================================================================
// Helper: Compare device view with host data
// ============================================================================

template <class T, class ExecSpace = Kokkos::DefaultExecutionSpace>
bool view_matches(Kokkos::View<T*, ExecSpace> device_view, const std::vector<T>& expected) {
  static_assert(
      Kokkos::SpaceAccessibility<typename ExecSpace::memory_space, Kokkos::HostSpace>::accessible,
      "View must be accessible from HostSpace or use mirror");

  const int n = device_view.extent(0);
  if (n != static_cast<int>(expected.size())) {
    return false;
  }

  auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_view);

  for (int i = 0; i < n; ++i) {
    if (host_view(i) != expected[i]) {
      return false;
    }
  }
  return true;
}

// Overload for const T (e.g., View<const int*, ExecSpace>)
template <class T, class ExecSpace = Kokkos::DefaultExecutionSpace>
bool view_matches(Kokkos::View<const T*, ExecSpace> device_view, const std::vector<T>& expected) {
  static_assert(
      Kokkos::SpaceAccessibility<typename ExecSpace::memory_space, Kokkos::HostSpace>::accessible,
      "View must be accessible from HostSpace or use mirror");

  const int n = device_view.extent(0);
  if (n != static_cast<int>(expected.size())) {
    return false;
  }

  auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_view);

  for (int i = 0; i < n; ++i) {
    if (host_view(i) != expected[i]) {
      return false;
    }
  }
  return true;
}

// ============================================================================
// Helper: Parallel reduce to compute sum
// ============================================================================

template <class ExecSpace = Kokkos::DefaultExecutionSpace>
int parallel_sum(int n) {
  int result = 0;
  Kokkos::parallel_reduce(
      "test_sum",
      n,
      KOKKOS_LAMBDA(int i, int& local_result) { local_result += i; },
      result);
  return result;
}

// ============================================================================
// Helper: Fill view with a pattern
// ============================================================================

template <class T, class ExecSpace = Kokkos::DefaultExecutionSpace>
void fill_with_index(Kokkos::View<T*, ExecSpace> view) {
  const int n = view.extent(0);
  Kokkos::parallel_for("fill_index", n, KOKKOS_LAMBDA(int i) { view(i) = static_cast<T>(i); });
  Kokkos::fence();
}

template <class T, class ExecSpace = Kokkos::DefaultExecutionSpace>
void fill_with_value(Kokkos::View<T*, ExecSpace> view, T value) {
  const int n = view.extent(0);
  Kokkos::parallel_for("fill_value", n, KOKKOS_LAMBDA(int i) { view(i) = value; });
  Kokkos::fence();
}

// ============================================================================
// Helper: Copy to/from host
// ============================================================================

template <class T, class ExecSpace = Kokkos::DefaultExecutionSpace>
std::vector<T> to_host_vector(Kokkos::View<const T*, ExecSpace> device_view) {
  const int n = device_view.extent(0);
  std::vector<T> host_data(n);

  auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_view);
  for (int i = 0; i < n; ++i) {
    host_data[i] = host_view(i);
  }

  return host_data;
}

template <class T, class ExecSpace = Kokkos::DefaultExecutionSpace>
Kokkos::View<T*, ExecSpace> to_device_view(const std::vector<T>& host_data) {
  Kokkos::View<T*, ExecSpace> device_view("device_view", host_data.size());
  auto host_mirror = Kokkos::create_mirror_view(device_view);

  for (size_t i = 0; i < host_data.size(); ++i) {
    host_mirror(i) = host_data[i];
  }

  Kokkos::deep_copy(device_view, host_mirror);
  return device_view;
}

// ============================================================================
// Helper: Device-side comparison
// ============================================================================

template <class T, class ExecSpace = Kokkos::DefaultExecutionSpace>
int count_mismatches(Kokkos::View<const T*, ExecSpace> view, const std::vector<T>& expected) {
  const int n = view.extent(0);
  int mismatches = 0;

  // Copy expected to device
  Kokkos::View<T*, ExecSpace> expected_device("expected", n);
  auto expected_host = Kokkos::create_mirror_view(expected_device);
  for (int i = 0; i < n; ++i) {
    expected_host(i) = expected[i];
  }
  Kokkos::deep_copy(expected_device, expected_host);

  Kokkos::parallel_reduce(
      "count_mismatches",
      n,
      KOKKOS_LAMBDA(int i, int& local_mismatches) {
        if (view(i) != expected_device(i)) {
          local_mismatches++;
        }
      },
      mismatches);

  return mismatches;
}

// Overload for non-const view
template <class T, class ExecSpace = Kokkos::DefaultExecutionSpace>
int count_mismatches(Kokkos::View<T*, ExecSpace> view, const std::vector<T>& expected) {
  // Just call the const version by casting
  return count_mismatches(Kokkos::View<const T*, ExecSpace>(view), expected);
}

// ============================================================================
// GTest macros for device views
// ============================================================================

#define EXPECT_VIEW_EQ(view, expected)                      \
  EXPECT_TRUE(subsetix::test::view_matches(view, expected)) \
      << "Device view does not match expected values"

#define ASSERT_VIEW_EQ(view, expected)                      \
  ASSERT_TRUE(subsetix::test::view_matches(view, expected)) \
      << "Device view does not match expected values"

// ============================================================================
// Test fixture base class
// ============================================================================

class KokkosTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Kokkos is already initialized in test_main.cpp
  }

  void TearDown() override {
    Kokkos::fence();  // Ensure all operations complete
  }

  // Get default execution space for testing
  static auto exec_space() { return Kokkos::DefaultExecutionSpace{}; }

  // Get host execution space
  static auto host_space() { return Kokkos::DefaultHostExecutionSpace{}; }
};

}  // namespace subsetix::test
