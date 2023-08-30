#include "benchmark/benchmark.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

template <typename PointTy, bool IsRandom>
void BM_Pippenger(benchmark::State& state) {
  PointTy::Curve::Init();
  MSMTestSet<PointTy> test_set;
  if constexpr (IsRandom) {
    test_set = MSMTestSet<PointTy>::Random(state.range(0), MSMMethod::kNone);
  } else {
    test_set =
        MSMTestSet<PointTy>::NonUniform(state.range(0), 10, MSMMethod::kNone);
  }
  Pippenger<PointTy> pippenger;
  using ReturnTy = typename Pippenger<PointTy>::ReturnTy;
  ReturnTy ret;
  for (auto _ : state) {
    pippenger.Run(test_set.bases.begin(), test_set.bases.end(),
                  test_set.scalars.begin(), test_set.scalars.end(), &ret);
  }
  benchmark::DoNotOptimize(ret);
}

template <typename PointTy>
void BM_PippengerRandom(benchmark::State& state) {
  BM_Pippenger<PointTy, true>(state);
}

template <typename PointTy>
void BM_PippengerNonUniform(benchmark::State& state) {
  BM_Pippenger<PointTy, false>(state);
}

BENCHMARK_TEMPLATE(BM_PippengerRandom, bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerNonUniform, bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);

}  // namespace tachyon::math

// clang-format off
// Executing tests from //tachyon/math/elliptic_curves/msm/algorithms:pippenger_benchmark
// -----------------------------------------------------------------------------
// 2023-08-30T05:54:01+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/2e01f4ccafa60589f9bbdbefc5d15e2a/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/elliptic_curves/msm/algorithms/pippenger_benchmark.runfiles/kroma_network_tachyon/tachyon/math/elliptic_curves/msm/algorithms/pippenger_benchmark
// Run on (32 X 5500 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 2.71, 2.64, 1.87
// -----------------------------------------------------------------------------------------------
// Benchmark                                                     Time             CPU   Iterations
// -----------------------------------------------------------------------------------------------
// BM_PippengerRandom<bn254::G1AffinePoint>/32768         34908874 ns     33396236 ns           21
// BM_PippengerRandom<bn254::G1AffinePoint>/65536         61684394 ns     55921674 ns           10
// BM_PippengerRandom<bn254::G1AffinePoint>/131072       109082434 ns     86647108 ns            9
// BM_PippengerRandom<bn254::G1AffinePoint>/262144       215575314 ns    148382267 ns            5
// BM_PippengerRandom<bn254::G1AffinePoint>/524288       410419226 ns    279332868 ns            3
// BM_PippengerRandom<bn254::G1AffinePoint>/1048576      755907536 ns    512204693 ns            1
// BM_PippengerNonUniform<bn254::G1AffinePoint>/32768     39093761 ns     36180793 ns           25
// BM_PippengerNonUniform<bn254::G1AffinePoint>/65536     57416010 ns     51653336 ns           10
// BM_PippengerNonUniform<bn254::G1AffinePoint>/131072   108107143 ns     90550465 ns            9
// BM_PippengerNonUniform<bn254::G1AffinePoint>/262144   205311680 ns    162177059 ns            5
// BM_PippengerNonUniform<bn254::G1AffinePoint>/524288   396333456 ns    260226808 ns            3
// BM_PippengerNonUniform<bn254::G1AffinePoint>/1048576  745496988 ns    504996463 ns            1
// clang-format on
