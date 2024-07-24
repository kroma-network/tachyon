#include "benchmark/benchmark.h"

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::math::polynomials::univariate {

constexpr size_t kMaxDegree = (size_t{1} << 32) - 1;
template <typename F>
void BM_SerialFFT(benchmark::State& state) {
  using Domain = UnivariateEvaluationDomain<F, kMaxDegree>;
  using Evals = typename Domain::Evals;
  using DensePoly = typename Domain::DensePoly;

  F::Init();

  std::unique_ptr<Domain> domain = Domain::Create((size_t{1} << 21) - 1);
  std::vector<Evals> v;
  v.reserve(state.range(0));
  for (int64_t i = 0; i < state.range(0); ++i) {
    v.emplace_back(Evals::Random(domain->size()));
  }

  for (auto _ : state) {
    std::vector<Evals> evals;
    std::vector<DensePoly> polys;
    evals.reserve(v.size());
    evals.reserve(v.size());
    for (Evals& e : v) polys.emplace_back(domain->IFFT(std::move(e)));
    for (DensePoly& p : polys) evals.emplace_back(domain->FFT(std::move(p)));
    benchmark::DoNotOptimize(polys);
    benchmark::DoNotOptimize(evals);
    v = std::move(evals);
  }
}

template <typename F>
void BM_BatchFFT(benchmark::State& state) {
  using Domain = UnivariateEvaluationDomain<F, kMaxDegree>;
  using Evals = typename Domain::Evals;
  using DensePoly = typename Domain::DensePoly;
  F::Init();

  std::unique_ptr<Domain> domain = Domain::Create((size_t{1} << 21) - 1);
  std::vector<Evals> v;
  v.reserve(state.range(0));
  for (int64_t i = 0; i < state.range(0); ++i) {
    v.emplace_back(Evals::Random(domain->size()));
  }

  for (auto _ : state) {
    std::vector<DensePoly> polys = domain->IFFT(std::move(v));
    std::vector<Evals> evals = domain->FFT(std::move(polys));
    benchmark::DoNotOptimize(polys);
    benchmark::DoNotOptimize(evals);
    v = std::move(evals);
  }
}

BENCHMARK_TEMPLATE(BM_SerialFFT, bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(3, 128)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_TEMPLATE(BM_BatchFFT, bn254::Fr)
    ->RangeMultiplier(2)
    ->Range(3, 128)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

}  // namespace tachyon::math::polynomials::univariate

// clang-format off
// 2024-08-02T07:29:41+00:00
// Run on (64 X 3300.66 MHz CPU s) (AWS g5.16xlarge instance)
// CPU Caches:
//   L1 Data 32 KiB (x32)
//   L1 Instruction 32 KiB (x32)
//   L2 Unified 512 KiB (x32)
//   L3 Unified 16384 KiB (x8)
// Load Average: 12.85, 15.10, 11.74
// ---------------------------------------------------------------------------------------------
// Benchmark                                                   Time             CPU   Iterations
// ---------------------------------------------------------------------------------------------
// BM_SerialFFT<bn254::Fr>/3/iterations:3/real_time          150 ms          150 ms            3
// BM_SerialFFT<bn254::Fr>/4/iterations:3/real_time          205 ms          205 ms            3
// BM_SerialFFT<bn254::Fr>/8/iterations:3/real_time          403 ms          403 ms            3
// BM_SerialFFT<bn254::Fr>/16/iterations:3/real_time         833 ms          825 ms            3
// BM_SerialFFT<bn254::Fr>/32/iterations:3/real_time        1649 ms         1648 ms            3
// BM_SerialFFT<bn254::Fr>/64/iterations:3/real_time        3307 ms         3304 ms            3
// BM_SerialFFT<bn254::Fr>/128/iterations:3/real_time       6537 ms         6533 ms            3
// BM_BatchFFT<bn254::Fr>/3/iterations:3/real_time           172 ms          166 ms            3
// BM_BatchFFT<bn254::Fr>/4/iterations:3/real_time           208 ms          208 ms            3
// BM_BatchFFT<bn254::Fr>/8/iterations:3/real_time           406 ms          406 ms            3
// BM_BatchFFT<bn254::Fr>/16/iterations:3/real_time          814 ms          814 ms            3
// BM_BatchFFT<bn254::Fr>/32/iterations:3/real_time         1633 ms         1633 ms            3
// BM_BatchFFT<bn254::Fr>/64/iterations:3/real_time         3266 ms         3266 ms            3
// BM_BatchFFT<bn254::Fr>/128/iterations:3/real_time        7165 ms         6976 ms            3
// clang-format on
