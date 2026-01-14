# ============================================================================
# Subsetix Kokkos Benchmarks
# ============================================================================

# Create the main benchmark executable
add_executable(subsetix_benchmark_main
  benchmark_main.cpp
  example_benchmark.cpp
  intersection_benchmark.cpp
)

# Link libraries
target_link_libraries(subsetix_benchmark_main
  PRIVATE
    subsetix::core
    benchmark::benchmark
    benchmark::benchmark_main
)

# Apply sanitizer flags if enabled
if(SUBSETIX_ENABLE_SANITIZERS AND DEFINED SUBSETIX_SANITIZER_COMPILE_FLAGS)
  target_compile_options(subsetix_benchmark_main PRIVATE ${SUBSETIX_SANITIZER_COMPILE_FLAGS})
  target_link_options(subsetix_benchmark_main PRIVATE ${SUBSETIX_SANITIZER_LINK_FLAGS})
endif()

# Include directories
target_include_directories(subsetix_benchmark_main
  PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
)

# Set compiler options for benchmarks
target_compile_options(subsetix_benchmark_main
  PRIVATE
    -O3
)
