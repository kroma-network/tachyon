#include "benchmark/benchmark.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

template <typename Point, bool IsRandom>
void BM_Pippenger(benchmark::State& state) {
  Point::Curve::Init();
  MSMTestSet<Point> test_set;
  if constexpr (IsRandom) {
    test_set = MSMTestSet<Point>::Random(state.range(0), MSMMethod::kNone);
  } else {
    test_set =
        MSMTestSet<Point>::NonUniform(state.range(0), 10, MSMMethod::kNone);
  }
  Pippenger<Point> pippenger;
  using Bucket = typename Pippenger<Point>::Bucket;
  Bucket ret;
  for (auto _ : state) {
    pippenger.Run(test_set.bases.begin(), test_set.bases.end(),
                  test_set.scalars.begin(), test_set.scalars.end(), &ret);
  }
  benchmark::DoNotOptimize(ret);
}

template <typename Point>
void BM_PippengerRandom(benchmark::State& state) {
  BM_Pippenger<Point, true>(state);
}

template <typename Point>
void BM_PippengerNonUniform(benchmark::State& state) {
  BM_Pippenger<Point, false>(state);
}

BENCHMARK_TEMPLATE(BM_PippengerRandom, bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerNonUniform, bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);

}  // namespace tachyon::math

// clang-format off
// Executing tests from //tachyon/math/elliptic_curves/msm/algorithms/pippenger:pippenger_benchmark
// -----------------------------------------------------------------------------
// 2023-09-13T00:54:30+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/2e01f4ccafa60589f9bbdbefc5d15e2a/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_benchmark.runfiles/kroma_network_tachyon/tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_benchmark
// Run on (32 X 5500 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 3.01, 1.86, 1.28
// -----------------------------------------------------------------------------------------------
// Benchmark                                                     Time             CPU   Iterations
// -----------------------------------------------------------------------------------------------
// BM_PippengerRandom<bn254::G1AffinePoint>/32768         22116457 ns     22115633 ns           28
// BM_PippengerRandom<bn254::G1AffinePoint>/65536         43687728 ns     40820899 ns           18
// BM_PippengerRandom<bn254::G1AffinePoint>/131072        78677741 ns     67452254 ns           11
// BM_PippengerRandom<bn254::G1AffinePoint>/262144       166095297 ns    128153516 ns            6
// BM_PippengerRandom<bn254::G1AffinePoint>/524288       321956635 ns    230632976 ns            3
// BM_PippengerRandom<bn254::G1AffinePoint>/1048576      621665239 ns    435303495 ns            2
// BM_PippengerNonUniform<bn254::G1AffinePoint>/32768     23061827 ns     23060417 ns           35
// BM_PippengerNonUniform<bn254::G1AffinePoint>/65536     43744180 ns     42149318 ns           18
// BM_PippengerNonUniform<bn254::G1AffinePoint>/131072    77002460 ns     67416707 ns           11
// BM_PippengerNonUniform<bn254::G1AffinePoint>/262144   161179503 ns    125391620 ns            6
// BM_PippengerNonUniform<bn254::G1AffinePoint>/524288   304928462 ns    212309880 ns            3
// BM_PippengerNonUniform<bn254::G1AffinePoint>/1048576  589349508 ns    424155770 ns            2
// clang-format on
