#include "benchmark/benchmark.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

template <typename PointTy, bool IsRandom, bool ClearCache>
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
  pippenger.SetClearCacheForTesting(ClearCache);
  using ReturnTy = typename Pippenger<PointTy>::ReturnTy;
  ReturnTy ret;
  for (auto _ : state) {
    pippenger.Run(test_set.bases.begin(), test_set.bases.end(),
                  test_set.scalars.begin(), test_set.scalars.end(), &ret);
  }
  benchmark::DoNotOptimize(ret);
}

template <typename PointTy>
void BM_PippengerRandomWithCache(benchmark::State& state) {
  BM_Pippenger<PointTy, true, false>(state);
}

template <typename PointTy>
void BM_PippengerRandom(benchmark::State& state) {
  BM_Pippenger<PointTy, true, true>(state);
}

template <typename PointTy>
void BM_PippengerNonUniform(benchmark::State& state) {
  BM_Pippenger<PointTy, false, false>(state);
}

template <typename PointTy>
void BM_PippengerNonUniformWithCache(benchmark::State& state) {
  BM_Pippenger<PointTy, false, true>(state);
}

BENCHMARK_TEMPLATE(BM_PippengerRandom, bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerRandomWithCache, bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerNonUniform, bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerNonUniformWithCache, bn254::G1AffinePoint)
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
// --------------------------------------------------------------------------------------------------------
// Benchmark                                                              Time             CPU   Iterations
// --------------------------------------------------------------------------------------------------------
// BM_PippengerRandom<bn254::G1AffinePoint>/32768                 285647273 ns    285638670 ns            2
// BM_PippengerRandom<bn254::G1AffinePoint>/65536                 526158810 ns    526138471 ns            1
// BM_PippengerRandom<bn254::G1AffinePoint>/131072               1010507822 ns   1009032618 ns            1
// BM_PippengerRandom<bn254::G1AffinePoint>/262144               1928324699 ns   1928262578 ns            1
// BM_PippengerRandom<bn254::G1AffinePoint>/524288               3531723976 ns   3528962397 ns            1
// BM_PippengerRandom<bn254::G1AffinePoint>/1048576              6836301804 ns   6835938130 ns            1
// BM_PippengerRandomWithCache<bn254::G1AffinePoint>/32768        285346270 ns    285329861 ns            2
// BM_PippengerRandomWithCache<bn254::G1AffinePoint>/65536        556808710 ns    556777027 ns            1
// BM_PippengerRandomWithCache<bn254::G1AffinePoint>/131072      1006542206 ns   1006495662 ns            1
// BM_PippengerRandomWithCache<bn254::G1AffinePoint>/262144      1923613548 ns   1923518538 ns            1
// BM_PippengerRandomWithCache<bn254::G1AffinePoint>/524288      3514778852 ns   3514655273 ns            1
// BM_PippengerRandomWithCache<bn254::G1AffinePoint>/1048576     6841082573 ns   6840623047 ns            1
// BM_PippengerNonUniform<bn254::G1AffinePoint>/32768             277929703 ns    277919829 ns            3
// BM_PippengerNonUniform<bn254::G1AffinePoint>/65536             505287886 ns    505261371 ns            1
// BM_PippengerNonUniform<bn254::G1AffinePoint>/131072            987757206 ns    987699281 ns            1
// BM_PippengerNonUniform<bn254::G1AffinePoint>/262144           1875749111 ns   1875631566 ns            1
// BM_PippengerNonUniform<bn254::G1AffinePoint>/524288           3408781767 ns   3408557514 ns            1
// BM_PippengerNonUniform<bn254::G1AffinePoint>/1048576          6684160471 ns   6683772112 ns            1
// BM_PippengerNonUniformWithCache<bn254::G1AffinePoint>/32768    277943452 ns    277933102 ns            3
// BM_PippengerNonUniformWithCache<bn254::G1AffinePoint>/65536    509582043 ns    509548690 ns            1
// BM_PippengerNonUniformWithCache<bn254::G1AffinePoint>/131072  1011629820 ns   1011588293 ns            1
// BM_PippengerNonUniformWithCache<bn254::G1AffinePoint>/262144  1912229538 ns   1912126362 ns            1
// BM_PippengerNonUniformWithCache<bn254::G1AffinePoint>/524288  3389491558 ns   3389264901 ns            1
// BM_PippengerNonUniformWithCache<bn254::G1AffinePoint>/1048576 6666882277 ns   6666451543 ns            1
// clang-format on
