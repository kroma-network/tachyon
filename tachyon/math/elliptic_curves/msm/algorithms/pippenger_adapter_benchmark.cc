#include "benchmark/benchmark.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/test/msm_test_set.h"

namespace tachyon::math {

template <typename PointTy, bool IsRandom,
          enum PippengerParallelStrategy Strategy>
void BM_PippengerAdapter(benchmark::State& state) {
  PointTy::Curve::Init();
  MSMTestSet<PointTy> test_set;
  if constexpr (IsRandom) {
    test_set = MSMTestSet<PointTy>::Random(state.range(0), MSMMethod::kNone);
  } else {
    test_set =
        MSMTestSet<PointTy>::NonUniform(state.range(0), 10, MSMMethod::kNone);
  }
  PippengerAdapter<PointTy> pippenger;
  using Bucket = typename PippengerAdapter<PointTy>::Bucket;
  Bucket ret;
  for (auto _ : state) {
    pippenger.RunWithStrategy(test_set.bases.begin(), test_set.bases.end(),
                              test_set.scalars.begin(), test_set.scalars.end(),
                              Strategy, &ret);
  }
  benchmark::DoNotOptimize(ret);
}

template <typename PointTy>
void BM_PippengerAdapterRandomWithParallelWindow(benchmark::State& state) {
  BM_PippengerAdapter<PointTy, true,
                      PippengerParallelStrategy::kParallelWindow>(state);
}

template <typename PointTy>
void BM_PippengerAdapterNonUniformWithParallelWindow(benchmark::State& state) {
  BM_PippengerAdapter<PointTy, false,
                      PippengerParallelStrategy::kParallelWindow>(state);
}

template <typename PointTy>
void BM_PippengerAdapterRandomWithParallelTerm(benchmark::State& state) {
  BM_PippengerAdapter<PointTy, true, PippengerParallelStrategy::kParallelTerm>(
      state);
}

template <typename PointTy>
void BM_PippengerAdapterNonUniformWithParallelTerm(benchmark::State& state) {
  BM_PippengerAdapter<PointTy, false, PippengerParallelStrategy::kParallelTerm>(
      state);
}

template <typename PointTy>
void BM_PippengerAdapterRandomWithParallelWindowAndTerm(
    benchmark::State& state) {
  BM_PippengerAdapter<PointTy, true,
                      PippengerParallelStrategy::kParallelWindowAndTerm>(state);
}

template <typename PointTy>
void BM_PippengerAdapterNonUniformWithParallelWindowAndTerm(
    benchmark::State& state) {
  BM_PippengerAdapter<PointTy, false,
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
// Executing tests from //tachyon/math/elliptic_curves/msm/algorithms:pippenger_adapter_benchmark
// -----------------------------------------------------------------------------
// 2023-08-31T01:41:00+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/2e01f4ccafa60589f9bbdbefc5d15e2a/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/elliptic_curves/msm/algorithms/pippenger_adapter_benchmark.runfiles/kroma_network_tachyon/tachyon/math/elliptic_curves/msm/algorithms/pippenger_adapter_benchmark
// Run on (32 X 5500 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 5.07, 3.03, 2.55
// -------------------------------------------------------------------------------------------------------------------------------
// Benchmark                                                                                     Time             CPU   Iterations
// -------------------------------------------------------------------------------------------------------------------------------
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/32768                36152965 ns     34685080 ns           21
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/65536                63747905 ns     59473935 ns           11
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/131072              118090590 ns     97648608 ns            6
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/262144              219497681 ns    153230226 ns            5
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/524288              413156907 ns    323098991 ns            3
// BM_PippengerAdapterRandomWithParallelWindow<bn254::G1AffinePoint>/1048576             779128075 ns    527287631 ns            1
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/32768            33103685 ns     32278959 ns           24
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/65536            59113836 ns     53515227 ns           10
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/131072          105071200 ns     88573100 ns            9
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/262144          211070061 ns    154581471 ns            5
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/524288          430753072 ns    355536668 ns            3
// BM_PippengerAdapterNonUniformWithParallelWindow<bn254::G1AffinePoint>/1048576         758098364 ns    500099233 ns            1
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/32768                  37594805 ns     32610251 ns           24
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/65536                  71606259 ns     55258869 ns           12
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/131072                107148111 ns     96393369 ns            8
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/262144                212497592 ns    174183040 ns            4
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/524288                394236803 ns    317781782 ns            2
// BM_PippengerAdapterRandomWithParallelTerm<bn254::G1AffinePoint>/1048576               643808842 ns    525789672 ns            1
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/32768              40219138 ns     32314687 ns           24
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/65536              56364516 ns     53881628 ns           12
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/131072            100064516 ns     89066432 ns            8
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/262144            186987877 ns    167513476 ns            4
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/524288            339649677 ns    306251663 ns            2
// BM_PippengerAdapterNonUniformWithParallelTerm<bn254::G1AffinePoint>/1048576           618226528 ns    548263127 ns            1
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/32768        159804106 ns    159796485 ns            4
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/65536        293185234 ns    293157717 ns            2
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/131072       541335583 ns    534124399 ns            1
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/262144      1050300837 ns   1029148338 ns            1
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/524288      1970518589 ns   1928973412 ns            1
// BM_PippengerAdapterRandomWithParallelWindowAndTerm<bn254::G1AffinePoint>/1048576     3633013487 ns   3545470719 ns            1
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/32768    153255129 ns    153244278 ns            5
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/65536    286615849 ns    286605001 ns            2
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/131072   530231476 ns    525174415 ns            1
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/262144  1019241333 ns   1005476065 ns            1
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/524288  1995004177 ns   1959156505 ns            1
// BM_PippengerAdapterNonUniformWithParallelWindowAndTerm<bn254::G1AffinePoint>/1048576 3475970268 ns   3417240024 ns            1
// clang-format on
