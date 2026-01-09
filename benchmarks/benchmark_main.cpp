#include <Kokkos_Core.hpp>

#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  // Initialize Kokkos before running benchmarks
  Kokkos::initialize(argc, argv);

  // Initialize Google Benchmark
  ::benchmark::Initialize(&argc, argv);

  // Run benchmarks (Google Benchmark handles flags automatically)
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    // No unrecognized arguments - this is normal behavior
  }

  // Always run benchmarks (Google Benchmark will skip if appropriate flags are set)
  ::benchmark::RunSpecifiedBenchmarks();

  // Shutdown Google Benchmark
  ::benchmark::Shutdown();

  // Finalize Kokkos after benchmarks complete
  Kokkos::finalize();

  return 0;
}
