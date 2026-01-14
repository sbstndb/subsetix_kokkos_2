# ============================================================================
# Subsetix Kokkos Tests
# ============================================================================

# Create the main test executable
add_executable(subsetix_test_main
  test_main.cpp
  example_test.cpp
  intersection_test.cpp
)

# Link libraries
target_link_libraries(subsetix_test_main
  PRIVATE
    subsetix::core
    gtest_main
)

# Apply sanitizer flags if enabled
if(SUBSETIX_ENABLE_SANITIZERS AND DEFINED SUBSETIX_SANITIZER_COMPILE_FLAGS)
  target_compile_options(subsetix_test_main PRIVATE ${SUBSETIX_SANITIZER_COMPILE_FLAGS})
  target_link_options(subsetix_test_main PRIVATE ${SUBSETIX_SANITIZER_LINK_FLAGS})
endif()

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
