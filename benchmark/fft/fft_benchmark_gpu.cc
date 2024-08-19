#if TACHYON_CUDA
#include <iostream>

// clang-format off
#include "benchmark/fft/fft_config.h"
#include "benchmark/fft/fft_runner.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::benchmark {

using namespace math;

template <typename Domain, typename PolyOrEvals,
          typename RetPoly = std::conditional_t<
              std::is_same_v<PolyOrEvals, typename Domain::Evals>,
              typename Domain::DensePoly, typename Domain::Evals>>
void Run(const FFTConfig& config) {
  using F = typename Domain::Field;

  std::string_view name;
  if (config.run_ifft()) {
    name = "IFFT Benchmark GPU";
  } else {
    name = "FFT Benchmark GPU";
  }

  SimpleReporter reporter(name);

  reporter.SetXLabel("Degree (2ˣ)");
  reporter.SetColumnLabels(base::Map(config.exponents(), [](uint32_t exponent) {
    return base::NumberToString(exponent);
  }));

  std::vector<size_t> degrees = config.GetDegrees();

  std::cout << "Generating evaluation domain and random polys..." << std::endl;
  std::vector<std::unique_ptr<Domain>> domains =
      base::Map(degrees, [](size_t degree) { return Domain::Create(degree); });
  std::vector<PolyOrEvals> polys = base::Map(
      degrees, [](size_t degree) { return PolyOrEvals::Random(degree); });
  std::cout << "Generation completed" << std::endl;

  IcicleNTTHolder<F> icicle_ntt_holder = IcicleNTTHolder<F>::Create();
  CHECK(icicle_ntt_holder->Init(domains.back()->group_gen()));

  FFTRunner<Domain, PolyOrEvals> runner(reporter);
  runner.set_polys(polys);
  runner.set_domains(absl::MakeSpan(domains));

  std::vector<RetPoly> results;
  std::vector<RetPoly> results_gpu;
  bool kShouldRecord = true;
  if constexpr (std::is_same_v<PolyOrEvals, typename Domain::Evals>) {
    runner.Run(Vendor::TachyonCPU(),
               tachyon_bn254_univariate_evaluation_domain_ifft_inplace, degrees,
               results, kShouldRecord);
    runner.SwitchToIcicle(&icicle_ntt_holder);
    runner.Run(Vendor::TachyonGPU(),
               tachyon_bn254_univariate_evaluation_domain_ifft_inplace, degrees,
               results_gpu, kShouldRecord);
    // NOLINTNEXTLINE(readability/braces)
  } else if constexpr (std::is_same_v<PolyOrEvals,
                                      typename Domain::DensePoly>) {
    runner.Run(Vendor::TachyonCPU(),
               tachyon_bn254_univariate_evaluation_domain_fft_inplace, degrees,
               results, kShouldRecord);
    runner.SwitchToIcicle(&icicle_ntt_holder);
    runner.Run(Vendor::TachyonGPU(),
               tachyon_bn254_univariate_evaluation_domain_fft_inplace, degrees,
               results_gpu, kShouldRecord);
  }
  if (config.check_results()) {
    CHECK(results == results_gpu) << "Results not matched";
  }

  reporter.Show();
}

int RealMain(int argc, char** argv) {
  using Field = bn254::Fr;
  constexpr size_t kMaxDegree = SIZE_MAX - 1;
  using Domain = UnivariateEvaluationDomain<Field, kMaxDegree>;
  using DensePoly = Domain::DensePoly;
  using Evals = Domain::Evals;

  Field::Init();

  FFTConfig config;
  FFTConfig::Options options;
  if (!config.Parse(argc, argv, options)) {
    return 1;
  }

  if (config.run_ifft()) {
    Run<Domain, Evals>(config);
  } else {
    Run<Domain, DensePoly>(config);
  }

  return 0;
}

}  // namespace tachyon::benchmark

int main(int argc, char** argv) {
  return tachyon::benchmark::RealMain(argc, argv);
}
#else
#include "tachyon/base/console/iostream.h"

int main(int argc, char **argv) {
  tachyon_cerr << "please build with --config cuda" << std::endl;
  return 1;
}
#endif  // TACHYON_CUDA
