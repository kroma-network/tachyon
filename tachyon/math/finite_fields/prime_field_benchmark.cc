#include "absl/base/call_once.h"
#include "benchmark/benchmark.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks.h"

constexpr const size_t kTestNum = 1000;

namespace tachyon::math {
namespace {

template <typename PrimeFieldTy>
static std::vector<PrimeFieldTy>& PrepareTestSet() {
  static base::NoDestructor<std::vector<PrimeFieldTy>> test_set;
  static absl::once_flag once;
  absl::call_once(once, []() {
    test_set->reserve(kTestNum);
    for (size_t i = 0; i < kTestNum; ++i) {
      test_set->push_back(PrimeFieldTy::Random());
    }
  });
  return *test_set;
}

}  // namespace

#define ADD_BENCHMARK(method, operator)                                       \
  template <typename PrimeFieldType>                                          \
  void BM_##method(benchmark::State& state) {                                 \
    using Config = typename PrimeFieldType::Config;                           \
    PrimeFieldType::Init();                                                   \
    auto& test_set = PrepareTestSet<PrimeField<Config>>();                    \
    std::vector<PrimeFieldType> converted_test_set;                           \
    converted_test_set.reserve(kTestNum);                                     \
    for (const auto& f : test_set) {                                          \
      converted_test_set.push_back(PrimeFieldType::FromBigInt(f.ToBigInt())); \
    }                                                                         \
    PrimeFieldType ret = PrimeFieldType::One();                               \
    size_t i = 0;                                                             \
    for (auto _ : state) {                                                    \
      ret operator##= converted_test_set[(i++) % kTestNum];                   \
    }                                                                         \
    benchmark::DoNotOptimize(ret);                                            \
  }

ADD_BENCHMARK(Add, +)
ADD_BENCHMARK(Mul, *)

#undef ADD_BENCHMARK

BENCHMARK_TEMPLATE(BM_Add, bn254::Fq)->Arg(kTestNum);
BENCHMARK_TEMPLATE(BM_Mul, bn254::Fq)->Arg(kTestNum);
#if defined(TACHYON_GMP_BACKEND)
BENCHMARK_TEMPLATE(BM_Add, bn254::FqGmp)->Arg(kTestNum);
BENCHMARK_TEMPLATE(BM_Mul, bn254::FqGmp)->Arg(kTestNum);
#endif  // defined(TACHYON_GMP_BACKEND)

BENCHMARK_TEMPLATE(BM_Add, Goldilocks)->Arg(kTestNum);
BENCHMARK_TEMPLATE(BM_Mul, Goldilocks)->Arg(kTestNum);
#if defined(TACHYON_GMP_BACKEND)
BENCHMARK_TEMPLATE(BM_Add, GoldilocksGmp)->Arg(kTestNum);
BENCHMARK_TEMPLATE(BM_Mul, GoldilocksGmp)->Arg(kTestNum);
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace tachyon::math

// clang-format off
// Executing tests from //tachyon/math/finite_fields:prime_field_benchmark
// -----------------------------------------------------------------------------
// 2023-07-31T03:20:30+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/2e01f4ccafa60589f9bbdbefc5d15e2a/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/finite_fields/prime_field_benchmark.runfiles/kroma_network_tachyon/tachyon/math/finite_fields/prime_field_benchmark
// Run on (32 X 5500 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 0.29, 1.32, 1.43
// ---------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations
// ---------------------------------------------------------------------
// BM_Add<bn254::Fq>/1000           6.11 ns         6.11 ns    114402345
// BM_Mul<bn254::Fq>/1000           22.4 ns         22.4 ns     31335557
// BM_Add<bn254::FqGmp>/1000        37.3 ns         37.3 ns     18789515
// BM_Mul<bn254::FqGmp>/1000        58.9 ns         58.9 ns     11830293
// BM_Add<Goldilocks>/1000         0.684 ns        0.684 ns   1000000000
// BM_Mul<Goldilocks>/1000          2.09 ns         2.09 ns    336274472
// BM_Add<GoldilocksGmp>/1000       28.2 ns         28.2 ns     24716406
// BM_Mul<GoldilocksGmp>/1000       26.7 ns         26.7 ns     26150948
// clang-format on
