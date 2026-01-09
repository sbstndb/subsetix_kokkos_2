#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

// Example test that verifies Kokkos is working correctly
TEST(KokkosInitialization, BasicTest) {
  // This test verifies that Kokkos was properly initialized
  EXPECT_TRUE(Kokkos::is_initialized());
}

// Helper function for parallel reduce test (needed for CUDA compatibility)
int compute_sum(int n) {
  int result = 0;
  Kokkos::parallel_reduce("simple_reduce", n,
    KOKKOS_LAMBDA(const int i, int& local_result) {
      local_result += i;
    }, result);
  return result;
}

// Example parallel for test
TEST(KokkosParallelFor, SimpleRange) {
  const int n = 100;

  // Sum of 0 to 99 should be 4950
  EXPECT_EQ(compute_sum(n), 4950);
}
