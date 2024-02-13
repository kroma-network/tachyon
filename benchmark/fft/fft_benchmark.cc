#include <iostream>

// clang-format off
#include "benchmark/fft/fft_config.h"
#include "benchmark/fft/fft_runner.h"
#include "benchmark/fft/simple_fft_benchmark_reporter.h"
// clang-format on
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon {

using namespace math;

extern "C" tachyon_bn254_fr* run_fft_halo2(const tachyon_bn254_fr* coeffs,
                                           size_t n,
                                           const tachyon_bn254_fr* omega,
                                           uint32_t k,
                                           uint64_t* duration_in_us);

int RealMain(int argc, char** argv) {
  using FFTRunner = FFTRunner<bn254::Fr>;
  using Domain = FFTRunner::Domain;
  using DensePoly = FFTRunner::DensePoly;
  FFTConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  bn254::Fr::Init();

  SimpleFFTBenchmarkReporter reporter("FFT Benchmark", config.k());
  reporter.AddVendor("halo2");

  FFTRunner runner(&reporter);

  std::cout << "Generating random poly..." << std::endl;
  uint64_t k = config.k();
  size_t n = size_t{1} << k;
  std::unique_ptr<Domain> domain = Domain::Create(n);
  DensePoly poly = DensePoly::Random(domain->size());
  std::cout << "Generation completed" << std::endl;

  runner.SetInput(&poly, std::move(domain));
  runner.Run(tachyon_bn254_univariate_evaluation_domain_fft);
  runner.RunExternal(run_fft_halo2);

  reporter.Show();

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
