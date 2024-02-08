#include "benchmark/benchmark.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

namespace tachyon::math {

template <typename F>
void BM_BatchInverse(benchmark::State& state) {
  using BigInt = typename F::BigIntTy;

  std::vector<F> fields = base::CreateVector(
      state.range(0), [](size_t i) { return F::FromBigInt(BigInt(i)); });
  for (auto _ : state) {
    CHECK(F::BatchInverseInPlace(fields));
  }
  benchmark::DoNotOptimize(fields);
}

BENCHMARK_TEMPLATE(BM_BatchInverse, bn254::Fq)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);

}  // namespace tachyon::math

// clang-format off
// Executing tests from //tachyon/math/base:batch_inverse_benchmark
// -----------------------------------------------------------------------------
// 2023-10-11T12:00:27+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/2e01f4ccafa60589f9bbdbefc5d15e2a/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/base/batch_inverse_benchmark.runfiles/kroma_network_tachyon/tachyon/math/base/batch_inverse_benchmark
// Run on (32 X 5500 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 5.03, 2.47, 1.98
// -----------------------------------------------------------------------------
// Benchmark                                   Time             CPU   Iterations
// -----------------------------------------------------------------------------
// BM_BatchInverse<bn254::Fq>/32768       171015 ns       170921 ns         4026
// BM_BatchInverse<bn254::Fq>/65536       496951 ns       477518 ns         1814
// BM_BatchInverse<bn254::Fq>/131072     1048433 ns      1014639 ns          689
// BM_BatchInverse<bn254::Fq>/262144     2118436 ns      2118300 ns          323
// BM_BatchInverse<bn254::Fq>/524288     4223493 ns      4220959 ns          222
// BM_BatchInverse<bn254::Fq>/1048576    7400200 ns      7321782 ns          125
// clang-format on
