#include "benchmark/benchmark.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

template <typename Point, bool IsRandom,
          enum PippengerParallelStrategy Strategy>
void BM_PippengerAdapter(benchmark::State& state) {
  Point::Curve::Init();
  MSMTestSet<Point> test_set;
  if constexpr (IsRandom) {
    test_set = MSMTestSet<Point>::Random(state.range(0), MSMMethod::kNone);
  } else {
    test_set =
        MSMTestSet<Point>::NonUniform(state.range(0), 10, MSMMethod::kNone);
  }
  PippengerAdapter<Point> pippenger;
  using Bucket = typename PippengerAdapter<Point>::Bucket;
  Bucket ret;
  for (auto _ : state) {
    pippenger.RunWithStrategy(test_set.bases.begin(), test_set.bases.end(),
                              test_set.scalars.begin(), test_set.scalars.end(),
                              Strategy, &ret);
  }
  benchmark::DoNotOptimize(ret);
}

template <typename Point>
void BM_PippengerAdapterRandomWithParallelWindow(benchmark::State& state) {
  BM_PippengerAdapter<Point, true, PippengerParallelStrategy::kParallelWindow>(
      state);
}

template <typename Point>
void BM_PippengerAdapterNonUniformWithParallelWindow(benchmark::State& state) {
  BM_PippengerAdapter<Point, false, PippengerParallelStrategy::kParallelWindow>(
      state);
}

template <typename Point>
void BM_PippengerAdapterRandomWithParallelTerm(benchmark::State& state) {
  BM_PippengerAdapter<Point, true, PippengerParallelStrategy::kParallelTerm>(
      state);
}

template <typename Point>
void BM_PippengerAdapterNonUniformWithParallelTerm(benchmark::State& state) {
  BM_PippengerAdapter<Point, false, PippengerParallelStrategy::kParallelTerm>(
      state);
}

template <typename Point>
void BM_PippengerAdapterRandomWithParallelWindowAndTerm(
    benchmark::State& state) {
  BM_PippengerAdapter<Point, true,
                      PippengerParallelStrategy::kParallelWindowAndTerm>(state);
}

template <typename Point>
void BM_PippengerAdapterNonUniformWithParallelWindowAndTerm(
    benchmark::State& state) {
  BM_PippengerAdapter<Point, false,
                      PippengerParallelStrategy::kParallelWindowAndTerm>(state);
}

BENCHMARK_TEMPLATE(BM_PippengerAdapterRandomWithParallelWindow,
                   bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerAdapterNonUniformWithParallelWindow,
                   bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerAdapterRandomWithParallelTerm,
                   bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerAdapterNonUniformWithParallelTerm,
                   bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerAdapterRandomWithParallelWindowAndTerm,
                   bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);
BENCHMARK_TEMPLATE(BM_PippengerAdapterNonUniformWithParallelWindowAndTerm,
                   bn254::G1AffinePoint)
    ->RangeMultiplier(2)
    ->Range(1 << 15, 1 << 20);

}  // namespace tachyon::math

// clang-format off
// Executing tests from //tachyon/math/elliptic_curves/msm/algorithms/pippenger:pippenger_adapter_benchmark
// -----------------------------------------------------------------------------
// 2023-09-13T01:54:29+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/2e01f4ccafa60589f9bbdbefc5d15e2a/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_adapter_benchmark.runfiles/kroma_network_tachyon/tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_adapter_benchmark
// Run on (32 X 5500 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 0.58, 0.51, 0.53
// -------------------------------------------------------------------------------------------------------------------------------
// Benchmark                                                                                     Time             CPU   Iterations
// -------------------------------------------------------------------------------------------------------------------------------
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/32768                23141976 ns     23140942 ns           27
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/65536                45281116 ns     44897489 ns           17
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/131072               79303765 ns     67719233 ns           10
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/262144              163616037 ns    125332664 ns            5
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/524288              321303527 ns    222457503 ns            3
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/1048576             628253341 ns    454716086 ns            2
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/32768            22177564 ns     22176387 ns           27
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/65536            43015189 ns     40323960 ns           18
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/131072           76032487 ns     64443376 ns           11
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/262144          161086917 ns    121884608 ns            6
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/524288          304606597 ns    222545666 ns            3
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/1048576         588236094 ns    438750185 ns            2
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/32768                  23018500 ns     22043368 ns           34
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/65536                  39519111 ns     38539498 ns           18
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/131072                 74930620 ns     67770503 ns           10
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/262144                137890458 ns    130253880 ns            6
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/524288                246021986 ns    230435266 ns            3
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/1048576               472836614 ns    416560205 ns            2
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/32768              19471671 ns     19470170 ns           37
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/65536              37021720 ns     36100273 ns           20
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/131072             71205117 ns     68483757 ns           11
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/262144            136134505 ns    115398568 ns            6
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/524288            241173903 ns    225617420 ns            3
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/1048576           439469576 ns    403132290 ns            2
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/32768        127219359 ns    127210795 ns            6
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/65536        231521209 ns    231510773 ns            3
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/131072       426907539 ns    426884276 ns            2
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/262144       830853224 ns    830388088 ns            1
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/524288      1585369587 ns   1580798267 ns            1
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/1048576     2858370543 ns   2855271982 ns            1
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/32768    123564482 ns    123560265 ns            6
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/65536    229732513 ns    229726383 ns            3
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/131072   417522907 ns    417503583 ns            2
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/262144   818992138 ns    818960598 ns            1
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/524288  1545445681 ns   1545361484 ns            1
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/1048576 2759630680 ns   2759482455 ns            1
// clang-format on
