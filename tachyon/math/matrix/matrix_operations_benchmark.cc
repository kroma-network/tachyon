#include "benchmark/benchmark.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/matrix/matrix_operations.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::math {

template <typename F>
void BM_MulMatVecNaive(benchmark::State& state) {
  F::Init();
  Matrix<F> matrix = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  Vector<F> vector = Vector<F>::Constant(state.range(0), F(5));
  for (auto _ : state) {
    vector = matrix * vector;
  }
  benchmark::DoNotOptimize(vector);
}

template <typename F>
void BM_MulMatVecSerial(benchmark::State& state) {
  F::Init();
  Matrix<F> matrix = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  Vector<F> vector = Vector<F>::Constant(state.range(0), F(5));
  for (auto _ : state) {
    vector = MulMatVecSerial(matrix, vector);
  }
  benchmark::DoNotOptimize(vector);
}

template <typename F>
void BM_MulMatVec(benchmark::State& state) {
  F::Init();
  Matrix<F> matrix = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  Vector<F> vector = Vector<F>::Constant(state.range(0), F(5));
  for (auto _ : state) {
    vector = MulMatVec(matrix, vector);
  }
  benchmark::DoNotOptimize(vector);
}

template <typename F>
void BM_MulMatMatNaive(benchmark::State& state) {
  F::Init();
  Matrix<F> matrix = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  Matrix<F> matrix2 = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  for (auto _ : state) {
    matrix2 = matrix * matrix2;
  }
  benchmark::DoNotOptimize(matrix2);
}

template <typename F>
void BM_MulMatMatSerial(benchmark::State& state) {
  F::Init();
  Matrix<F> matrix = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  Matrix<F> matrix2 = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  for (auto _ : state) {
    matrix2 = MulMatMatSerial(matrix, matrix2);
  }
  benchmark::DoNotOptimize(matrix2);
}

template <typename F>
void BM_MulMatMat(benchmark::State& state) {
  F::Init();
  Matrix<F> matrix = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  Matrix<F> matrix2 = Matrix<F>::Constant(state.range(0), state.range(0), F(5));
  for (auto _ : state) {
    matrix2 = MulMatMat(matrix, matrix2);
  }
  benchmark::DoNotOptimize(matrix2);
}

BENCHMARK_TEMPLATE(BM_MulMatVecNaive, bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(BM_MulMatVecSerial, bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(BM_MulMatVec, bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(BM_MulMatMatNaive, bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 9);
BENCHMARK_TEMPLATE(BM_MulMatMatSerial, bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 9);
BENCHMARK_TEMPLATE(BM_MulMatMat, bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 9);

}  // namespace tachyon::math

// clang-format off
// Executing tests from //tachyon/math/matrix:matrix_operations_benchmark
// -----------------------------------------------------------------------------
// 2024-07-19T04:56:59+00:00
// Running /home/chokobole/.cache/bazel/_bazel_chokobole/234690e3562329d13f7f07caac03dae4/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/matrix/matrix_operations_benchmark.runfiles/kroma_network_tachyon/tachyon/math/matrix/matrix_operations_benchmark
// Run on (32 X 5500.06 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 1.19, 1.31, 1.18
// -----------------------------------------------------------------------------
// Benchmark                                   Time             CPU   Iterations
// -----------------------------------------------------------------------------
// BM_MulMatVecNaive<bn254::Fr>/2            123 ns          122 ns      5617150
// BM_MulMatVecNaive<bn254::Fr>/4            280 ns          280 ns      2497914
// BM_MulMatVecNaive<bn254::Fr>/8            859 ns          859 ns       816942
// BM_MulMatVecNaive<bn254::Fr>/16          3113 ns         3112 ns       225830
// BM_MulMatVecNaive<bn254::Fr>/32         11736 ns        11736 ns        59687
// BM_MulMatVecNaive<bn254::Fr>/64         46452 ns        46442 ns        15218
// BM_MulMatVecNaive<bn254::Fr>/128       193267 ns       193259 ns         3652
// BM_MulMatVecNaive<bn254::Fr>/256       769266 ns       769131 ns          913
// BM_MulMatVecNaive<bn254::Fr>/512      3055500 ns      3055356 ns          229
// BM_MulMatVecNaive<bn254::Fr>/1024    14999901 ns     14999210 ns           48
// BM_MulMatVecSerial<bn254::Fr>/2          55.1 ns         55.1 ns     12941369
// BM_MulMatVecSerial<bn254::Fr>/4           196 ns          196 ns      3668682
// BM_MulMatVecSerial<bn254::Fr>/8           740 ns          740 ns       954441
// BM_MulMatVecSerial<bn254::Fr>/16         2863 ns         2863 ns       242618
// BM_MulMatVecSerial<bn254::Fr>/32        11403 ns        11403 ns        61397
// BM_MulMatVecSerial<bn254::Fr>/64        46126 ns        46126 ns        15170
// BM_MulMatVecSerial<bn254::Fr>/128      187898 ns       187899 ns         3730
// BM_MulMatVecSerial<bn254::Fr>/256      763410 ns       763404 ns          919
// BM_MulMatVecSerial<bn254::Fr>/512     3586593 ns      3586489 ns          198
// BM_MulMatVecSerial<bn254::Fr>/1024   32458958 ns     32457299 ns           23
// BM_MulMatVec<bn254::Fr>/2                9752 ns         9231 ns       192896
// BM_MulMatVec<bn254::Fr>/4                4711 ns         4480 ns       256574
// BM_MulMatVec<bn254::Fr>/8                5064 ns         4990 ns       160772
// BM_MulMatVec<bn254::Fr>/16               8102 ns         6706 ns       100000
// BM_MulMatVec<bn254::Fr>/32               5156 ns         5014 ns       149700
// BM_MulMatVec<bn254::Fr>/64              21412 ns        19453 ns        66057
// BM_MulMatVec<bn254::Fr>/128            105175 ns        95884 ns        16822
// BM_MulMatVec<bn254::Fr>/256           1653757 ns      1462683 ns         8835
// BM_MulMatVec<bn254::Fr>/512           1968314 ns      1750242 ns         1000
// BM_MulMatVec<bn254::Fr>/1024          3888062 ns      3581259 ns          153
// BM_MulMatMatNaive<bn254::Fr>/2            145 ns          145 ns      4820759
// BM_MulMatMatNaive<bn254::Fr>/4            835 ns          835 ns       835765
// BM_MulMatMatNaive<bn254::Fr>/8           8772 ns         8771 ns        79835
// BM_MulMatMatNaive<bn254::Fr>/16         63928 ns        63833 ns        11102
// BM_MulMatMatNaive<bn254::Fr>/32        481063 ns       481060 ns         1455
// BM_MulMatMatNaive<bn254::Fr>/64        963437 ns       963423 ns          729
// BM_MulMatMatNaive<bn254::Fr>/128      6737850 ns      6255369 ns          234
// BM_MulMatMatNaive<bn254::Fr>/256     17992549 ns     17970574 ns           41
// BM_MulMatMatNaive<bn254::Fr>/512    146447301 ns    146146565 ns            4
// BM_MulMatMatSerial<bn254::Fr>/2           102 ns          102 ns      6888084
// BM_MulMatMatSerial<bn254::Fr>/4           750 ns          750 ns       934540
// BM_MulMatMatSerial<bn254::Fr>/8          5855 ns         5855 ns       119456
// BM_MulMatMatSerial<bn254::Fr>/16        46651 ns        46647 ns        14998
// BM_MulMatMatSerial<bn254::Fr>/32       371483 ns       371476 ns         1884
// BM_MulMatMatSerial<bn254::Fr>/64      3000850 ns      3000784 ns          234
// BM_MulMatMatSerial<bn254::Fr>/128    24536289 ns     24536348 ns           29
// BM_MulMatMatSerial<bn254::Fr>/256   201230526 ns    201217998 ns            4
// BM_MulMatMatSerial<bn254::Fr>/512  1850822449 ns   1850779270 ns            1
// BM_MulMatMat<bn254::Fr>/2                6833 ns         6833 ns       100000
// BM_MulMatMat<bn254::Fr>/4                6848 ns         6847 ns       146381
// BM_MulMatMat<bn254::Fr>/8                7575 ns         7575 ns        91743
// BM_MulMatMat<bn254::Fr>/16              33371 ns        32436 ns        48004
// BM_MulMatMat<bn254::Fr>/32              50228 ns        50148 ns        10431
// BM_MulMatMat<bn254::Fr>/64            2107477 ns      1953816 ns         1000
// BM_MulMatMat<bn254::Fr>/128          13232534 ns     10078648 ns           67
// BM_MulMatMat<bn254::Fr>/256          21831894 ns     20891728 ns           30
// BM_MulMatMat<bn254::Fr>/512         163877392 ns    156880803 ns            5
// clang-format on
