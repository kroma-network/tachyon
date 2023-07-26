#include "absl/base/call_once.h"
#include "benchmark/benchmark.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

constexpr const size_t kTestNum = 1000;

namespace tachyon {
namespace math {
namespace {

static std::vector<bn254::FqGmp>& PrepareTestSet() {
  static base::NoDestructor<std::vector<bn254::FqGmp>> test_set;
  static absl::once_flag once;
  bn254::FqGmp::Init();
  absl::call_once(once, []() {
    test_set->reserve(kTestNum);
    for (size_t i = 0; i < kTestNum; ++i) {
      test_set->push_back(bn254::FqGmp::Random());
    }
  });
  return *test_set;
}

}  // namespace

template <typename PrimeFieldType>
void BM_Mul(benchmark::State& state) {
  std::vector<bn254::FqGmp>& test_set = PrepareTestSet();
  std::vector<PrimeFieldType> converted_test_set;
  converted_test_set.reserve(kTestNum);
  for (const bn254::FqGmp& f : test_set) {
    converted_test_set.push_back(PrimeFieldType::FromBigInt(f.ToBigInt()));
  }
  PrimeFieldType ret = PrimeFieldType::One();
  size_t i = 0;
  for (auto _ : state) {
    ret *= converted_test_set[(i++) % kTestNum];
  }
}

BENCHMARK_TEMPLATE(BM_Mul, bn254::FqGmp)->Arg(kTestNum);
BENCHMARK_TEMPLATE(BM_Mul, bn254::Fq)->Arg(kTestNum);

}  // namespace math
}  // namespace tachyon

// clang-format off
// Executing tests from //tachyon/math/finite_fields:finite_field_mul_benchmark
// -----------------------------------------------------------------------------
// 2023-07-24T05:58:38+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/2e01f4ccafa60589f9bbdbefc5d15e2a/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/math/finite_fields/finite_field_mul_benchmark.runfiles/kroma_network_tachyon/tachyon/math/finite_fields/finite_field_mul_benchmark
// Run on (32 X 4299.94 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 0.74, 1.08, 1.09
// --------------------------------------------------------------------
// Benchmark                          Time             CPU   Iterations
// --------------------------------------------------------------------
// BM_Mul<bn254::FqGmp>/1000       77.5 ns         77.5 ns      9230747
// BM_Mul<bn254::Fq>/1000          28.9 ns         28.9 ns     24175968
// clang-format on
