#include "benchmark/benchmark.h"

#include "tachyon/base/parallelize.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::zk {

template <typename F>
void BM_Parallelize(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<F> vec =
      base::CreateVectorParallel(n, []() { return F::Random(); });
  for (auto _ : state) {
    base::Parallelize(vec, [](absl::Span<F> chunk) {
      for (F& v : chunk) {
        v.DoubleInPlace();
      }
    });
  }
  benchmark::DoNotOptimize(vec);
}

template <typename F>
void BM_ForLoop(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<F> vec =
      base::CreateVectorParallel(n, []() { return F::Random(); });
  for (auto _ : state) {
    OMP_PARALLEL_FOR(size_t i = 0; i < n; ++i) { vec[i].DoubleInPlace(); }
  }
  benchmark::DoNotOptimize(vec);
}

BENCHMARK_TEMPLATE(BM_Parallelize, math::bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);

BENCHMARK_TEMPLATE(BM_ForLoop, math::bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);

}  // namespace tachyon::zk

// clang-format off
// Executing tests from //tachyon/zk/base:parallelize_benchmark
// -----------------------------------------------------------------------------
// 2024-02-08T13:37:00+00:00
// Running /home/chokobole/.cache/bazel/_bazel_chokobole/623c4ddaca6f9399eb551ba277ab230c/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/zk/base/parallelize_benchmark.runfiles/kroma_network_tachyon/tachyon/zk/base/parallelize_benchmark
// Run on (8 X 4600.18 MHz CPU s)
// CPU Caches:
//   L1 Data 32 KiB (x8)
//   L1 Instruction 32 KiB (x8)
//   L2 Unified 256 KiB (x8)
//   L3 Unified 12288 KiB (x1)
// Load Average: 2.21, 1.48, 1.54
// ----------------------------------------------------------------------------------
// Benchmark                                        Time             CPU   Iterations
// ----------------------------------------------------------------------------------
// BM_Parallelize<math::bn254::Fr>/32768        32137 ns        31658 ns        22342
// BM_Parallelize<math::bn254::Fr>/65536        64475 ns        64003 ns        11505
// BM_Parallelize<math::bn254::Fr>/131072      135352 ns       128462 ns         5805
// BM_Parallelize<math::bn254::Fr>/262144      348907 ns       336052 ns         2630
// BM_Parallelize<math::bn254::Fr>/524288      730779 ns       675620 ns         1322
// BM_Parallelize<math::bn254::Fr>/1048576    1993262 ns      1908117 ns          431
// BM_ForLoop<math::bn254::Fr>/32768            33624 ns        33551 ns        21413
// BM_ForLoop<math::bn254::Fr>/65536            66033 ns        65931 ns        10689
// BM_ForLoop<math::bn254::Fr>/131072          142503 ns       136927 ns         5561
// BM_ForLoop<math::bn254::Fr>/262144          284805 ns       269413 ns         2689
// BM_ForLoop<math::bn254::Fr>/524288          584841 ns       583951 ns         1369
// BM_ForLoop<math::bn254::Fr>/1048576        1621698 ns      1621422 ns          389
// clang-format on
