# ============================================================================
# Subsetix Kokkos Tests
# ============================================================================

# Create the main test executable
add_executable(subsetix_test_main
  test_main.cpp
  example_test.cpp
)

# Link libraries
target_link_libraries(subsetix_test_main
  PRIVATE
    subsetix::core
    gtest_main
)

# Include directories
target_include_directories(subsetix_test_main
  PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
)

# Register with CTest
add_test(NAME subsetix_test_main COMMAND subsetix_test_main)

# Set test properties
set_tests_properties(subsetix_test_main PROPERTIES
  LABELS "core"
)
