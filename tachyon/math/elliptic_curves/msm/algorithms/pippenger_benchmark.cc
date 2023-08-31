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
// 2023-08-31T23:50:45+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/d6800124b8b6155cc6ab653ae18dfdd6/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/elliptic_curves/msm/algorithms/pippenger_benchmark.runfiles/kroma_network_tachyon/tachyon/math/elliptic_curves/msm/algorithms/pippenger_benchmark
// Run on (32 X 5489.54 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 4.14, 3.58, 2.97
// -----------------------------------------------------------------------------------------------
// Benchmark                                                     Time             CPU   Iterations
// -----------------------------------------------------------------------------------------------
// BM_PippengerRandom<bn254::G1AffinePoint>/32768         34569606 ns     32515164 ns           23
// BM_PippengerRandom<bn254::G1AffinePoint>/65536         59088051 ns     49972215 ns           12
// BM_PippengerRandom<bn254::G1AffinePoint>/131072       104039362 ns     84752183 ns            7
// BM_PippengerRandom<bn254::G1AffinePoint>/262144       220377350 ns    158326024 ns            5
// BM_PippengerRandom<bn254::G1AffinePoint>/524288       405006965 ns    266241591 ns            3
// BM_PippengerRandom<bn254::G1AffinePoint>/1048576      793197632 ns    544881375 ns            1
// BM_PippengerNonUniform<bn254::G1AffinePoint>/32768     31510353 ns     29713886 ns           26
// BM_PippengerNonUniform<bn254::G1AffinePoint>/65536     55103302 ns     52560911 ns           10
// BM_PippengerNonUniform<bn254::G1AffinePoint>/131072   102975157 ns     89503428 ns            9
// BM_PippengerNonUniform<bn254::G1AffinePoint>/262144   208537149 ns    150291432 ns            5
// BM_PippengerNonUniform<bn254::G1AffinePoint>/524288   407595555 ns    267039153 ns            3
// BM_PippengerNonUniform<bn254::G1AffinePoint>/1048576  775470257 ns    513212445 ns            1
// clang-format on
