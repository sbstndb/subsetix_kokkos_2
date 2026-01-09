#include <vector>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

// Example test that verifies Kokkos is working correctly
TEST(KokkosInitialization, BasicTest) {
  // This test verifies that Kokkos was properly initialized
  EXPECT_TRUE(Kokkos::is_initialized());
}

// Helper function for parallel sum (CUDA-compatible)
int parallel_sum(int n) {
  int result = 0;
  Kokkos::parallel_reduce(
      "simple_reduce",
      n,
      KOKKOS_LAMBDA(int i, int& local_result) { local_result += i; },
      result);
  return result;
}

// Example parallel for test using helper function
TEST(KokkosParallelFor, SimpleRange) {
  const int n = 100;

  // Use the helper function to avoid CUDA compatibility issues
  int result = parallel_sum(n);

  // Sum of 0 to 99 should be 4950
  EXPECT_EQ(result, 4950);
}

// Helper function to fill view with index (CUDA-compatible)
void fill_with_index(Kokkos::View<int*, Kokkos::DefaultExecutionSpace> view) {
  const int n = view.extent(0);
  Kokkos::parallel_for("fill_index", n, KOKKOS_LAMBDA(int i) { view(i) = i; });
  Kokkos::fence();
}

// Example using view helpers
TEST(KokkosViewHelpers, FillAndCompare) {
  const int n = 10;

  // Create a device view
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace> device_view("test_view", n);

  // Fill with index values using helper
  fill_with_index(device_view);

  // Manual comparison (CUDA-compatible)
  auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_view);

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(host_view(i), i);
  }
}

// Helper function for sum with offset (CUDA-compatible)
int compute_sum_offset(int n, int offset) {
  int result = 0;
  Kokkos::parallel_reduce(
      "sum_offset",
      n,
      KOKKOS_LAMBDA(int i, int& local_result) { local_result += i + offset; },
      result);
  return result;
}

// Example using custom reduction
TEST(KokkosCustomReduce, SumWithOffset) {
  const int n = 5;
  const int offset = 10;

  // Compute sum of 10..14 = 60 using helper
  int result = compute_sum_offset(n, offset);

  EXPECT_EQ(result, 60);
}
