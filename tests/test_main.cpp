#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  // Initialize Kokkos before running tests
  Kokkos::initialize(argc, argv);

  // Initialize GoogleTest
  ::testing::InitGoogleTest(&argc, argv);

  // Run all tests
  const int result = RUN_ALL_TESTS();

  // Finalize Kokkos after tests complete
  Kokkos::finalize();

  return result;
}
