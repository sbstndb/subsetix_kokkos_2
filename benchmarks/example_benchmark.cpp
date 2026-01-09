#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>

// Example benchmark: parallel for
static void BM_ParallelFor(benchmark::State& state) {
  const int n = state.range(0);

  for (auto _ : state) {
    Kokkos::parallel_for("simple_for", n,
      KOKKOS_LAMBDA(const int i) {
        // Simple computation
        volatile double x = i * 2.0;
        (void)x;  // Prevent unused variable warning
      });
    Kokkos::fence();  // Ensure computation is complete
  }

  state.SetItemsProcessed(state.iterations() * n);
}

// Register the benchmark with different sizes
BENCHMARK(BM_ParallelFor)->Arg(1000)->Arg(10000)->Arg(100000);

// Example benchmark: parallel reduce
static void BM_ParallelReduce(benchmark::State& state) {
  const int n = state.range(0);

  for (auto _ : state) {
    int result = 0;
    Kokkos::parallel_reduce("simple_reduce", n,
      KOKKOS_LAMBDA(const int i, int& local_result) {
        local_result += i;
      }, result);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_ParallelReduce)->Arg(1000)->Arg(10000)->Arg(100000);
