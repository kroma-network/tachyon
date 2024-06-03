#include "benchmark/benchmark.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

namespace tachyon::math {

template <typename F>
void BM_BatchInverseSerial(benchmark::State& state) {
  using BigInt = typename F::BigIntTy;

  std::vector<F> fields = base::CreateVector(
      state.range(0), [](size_t i) { return F::FromBigInt(BigInt(i + 1)); });
  for (auto _ : state) {
    CHECK(F::BatchInverseInPlaceSerial(fields));
  }
  benchmark::DoNotOptimize(fields);
}

template <typename F>
void BM_BatchInverse(benchmark::State& state) {
  using BigInt = typename F::BigIntTy;

  std::vector<F> fields = base::CreateVector(
      state.range(0), [](size_t i) { return F::FromBigInt(BigInt(i + 1)); });
  for (auto _ : state) {
    CHECK(F::BatchInverseInPlace(fields));
  }
  benchmark::DoNotOptimize(fields);
}

template <typename F>
void BM_InverseParallelFor(benchmark::State& state) {
  using BigInt = typename F::BigIntTy;

  std::vector<F> fields = base::CreateVector(
      state.range(0), [](size_t i) { return F::FromBigInt(BigInt(i + 1)); });
  for (auto _ : state) {
    OPENMP_PARALLEL_FOR(size_t i = 0; i < fields.size(); ++i) {
      CHECK(fields[i].InverseInPlace());
    }
  }
  benchmark::DoNotOptimize(fields);
}

BENCHMARK_TEMPLATE(BM_BatchInverseSerial, bn254::Fq)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);

BENCHMARK_TEMPLATE(BM_BatchInverse, bn254::Fq)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);

BENCHMARK_TEMPLATE(BM_InverseParallelFor, bn254::Fq)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);

}  // namespace tachyon::math

// clang-format off
// Executing tests from //tachyon/math/base:batch_inverse_benchmark
// -----------------------------------------------------------------------------
// 2024-02-13T05:16:40+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/d6800124b8b6155cc6ab653ae18dfdd6/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/base/batch_inverse_benchmark.runfiles/kroma_network_tachyon/tachyon/math/base/batch_inverse_benchmark
// Run on (32 X 5500 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 12.51, 18.64, 11.34
// ------------------------------------------------------------------------------------
// Benchmark                                          Time             CPU   Iterations
// ------------------------------------------------------------------------------------
// BM_BatchInverseSerial<bn254::Fq>/32            3107 ns         3107 ns       225050
// BM_BatchInverseSerial<bn254::Fq>/64            5154 ns         5154 ns       135741
// BM_BatchInverseSerial<bn254::Fq>/128           8236 ns         8235 ns        84985
// BM_BatchInverseSerial<bn254::Fq>/256          14356 ns        14356 ns        48760
// BM_BatchInverseSerial<bn254::Fq>/512          26553 ns        26553 ns        26368
// BM_BatchInverseSerial<bn254::Fq>/1024         51145 ns        51144 ns        13658
// BM_BatchInverseSerial<bn254::Fq>/2048        100272 ns       100267 ns         6978
// BM_BatchInverseSerial<bn254::Fq>/4096        198646 ns       198616 ns         3521
// BM_BatchInverseSerial<bn254::Fq>/8192        405130 ns       404911 ns         1768
// BM_BatchInverseSerial<bn254::Fq>/16384       790625 ns       790610 ns          879
// BM_BatchInverseSerial<bn254::Fq>/32768      1581307 ns      1581250 ns          442
// BM_BatchInverseSerial<bn254::Fq>/65536      3160691 ns      3160623 ns          220
// BM_BatchInverseSerial<bn254::Fq>/131072     6327060 ns      6326715 ns          109
// BM_BatchInverseSerial<bn254::Fq>/262144    12804429 ns     12803991 ns           54
// BM_BatchInverseSerial<bn254::Fq>/524288    25865104 ns     25863830 ns           27
// BM_BatchInverseSerial<bn254::Fq>/1048576   60116311 ns     60114166 ns           12
// BM_BatchInverse<bn254::Fq>/32                  3112 ns         3111 ns       225093
// BM_BatchInverse<bn254::Fq>/64                  5211 ns         5210 ns       135043
// BM_BatchInverse<bn254::Fq>/128                 8259 ns         8259 ns        84689
// BM_BatchInverse<bn254::Fq>/256              1416243 ns      1257341 ns         1000
// BM_BatchInverse<bn254::Fq>/512               157228 ns       144531 ns        75464
// BM_BatchInverse<bn254::Fq>/1024             3287592 ns      2926591 ns         1000
// BM_BatchInverse<bn254::Fq>/2048               29447 ns        27628 ns        51731
// BM_BatchInverse<bn254::Fq>/4096              693075 ns       621306 ns        33576
// BM_BatchInverse<bn254::Fq>/8192             8797123 ns      7401274 ns           94
// BM_BatchInverse<bn254::Fq>/16384            8847607 ns      7432851 ns          107
// BM_BatchInverse<bn254::Fq>/32768            8931447 ns      7878056 ns           88
// BM_BatchInverse<bn254::Fq>/65536            9000511 ns      7951377 ns           84
// BM_BatchInverse<bn254::Fq>/131072           4666971 ns      4170687 ns          199
// BM_BatchInverse<bn254::Fq>/262144           6942271 ns      6849338 ns          335
// BM_BatchInverse<bn254::Fq>/524288          10203157 ns      9504026 ns          148
// BM_BatchInverse<bn254::Fq>/1048576          8687019 ns      7257869 ns          197
// BM_InverseParallelFor<bn254::Fq>/32           60951 ns        59605 ns        10000
// BM_InverseParallelFor<bn254::Fq>/64          110740 ns       101640 ns        66209
// BM_InverseParallelFor<bn254::Fq>/128        5896806 ns      5134363 ns         1000
// BM_InverseParallelFor<bn254::Fq>/256        8795295 ns      7209770 ns          114
// BM_InverseParallelFor<bn254::Fq>/512        8776199 ns      7490824 ns           88
// BM_InverseParallelFor<bn254::Fq>/1024       7800870 ns      6280526 ns          108
// BM_InverseParallelFor<bn254::Fq>/2048       9148910 ns      7553591 ns           81
// BM_InverseParallelFor<bn254::Fq>/4096       9734373 ns      8234960 ns          100
// BM_InverseParallelFor<bn254::Fq>/8192       8042700 ns      6609198 ns           93
// BM_InverseParallelFor<bn254::Fq>/16384      6758195 ns      5395180 ns          312
// BM_InverseParallelFor<bn254::Fq>/32768      6857435 ns      5830095 ns          108
// BM_InverseParallelFor<bn254::Fq>/65536     10401757 ns      9951429 ns           53
// BM_InverseParallelFor<bn254::Fq>/131072    20822048 ns     18544346 ns           36
// BM_InverseParallelFor<bn254::Fq>/262144    37196371 ns     34716896 ns           18
// BM_InverseParallelFor<bn254::Fq>/524288    76307365 ns     73958721 ns            7
// BM_InverseParallelFor<bn254::Fq>/1048576  154857337 ns    140068730 ns            4
// clang-format on
