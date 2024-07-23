#include <stddef.h>
#include <stdint.h>

#include <iostream>

// clang-format off
#include "benchmark/fft/fft_config.h"
#include "benchmark/fft/fft_runner.h"
#include "benchmark/fft/simple_fft_benchmark_reporter.h"
// clang-format on
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/elliptic_curves/bn/bn254/halo2/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon {

using namespace math;

extern "C" tachyon_bn254_fr* run_fft_arkworks(const tachyon_bn254_fr* coeffs,
                                              size_t n,
                                              const tachyon_bn254_fr* omega,
                                              uint32_t k,
                                              uint64_t* duration_in_us);

extern "C" tachyon_bn254_fr* run_ifft_arkworks(
    const tachyon_bn254_fr* coeffs, size_t n, const tachyon_bn254_fr* omega_inv,
    uint32_t k, uint64_t* duration_in_us);

extern "C" tachyon_bn254_fr* run_fft_bellman(const tachyon_bn254_fr* coeffs,
                                             size_t n,
                                             const tachyon_bn254_fr* omega,
                                             uint32_t k,
                                             uint64_t* duration_in_us);

extern "C" tachyon_bn254_fr* run_ifft_bellman(const tachyon_bn254_fr* coeffs,
                                              size_t n,
                                              const tachyon_bn254_fr* omega_inv,
                                              uint32_t k,
                                              uint64_t* duration_in_us);

extern "C" tachyon_bn254_fr* run_fft_halo2(const tachyon_bn254_fr* coeffs,
                                           size_t n,
                                           const tachyon_bn254_fr* omega,
                                           uint32_t k,
                                           uint64_t* duration_in_us);

extern "C" tachyon_bn254_fr* run_ifft_halo2(const tachyon_bn254_fr* coeffs,
                                            size_t n,
                                            const tachyon_bn254_fr* omega_inv,
                                            uint32_t k,
                                            uint64_t* duration_in_us);

template <typename PolyOrEvals>
void CheckResults(bool check_results, const std::vector<PolyOrEvals>& results,
                  const std::vector<PolyOrEvals>& results_vendor) {
  if (check_results) {
    CHECK(results == results_vendor) << "Results not matched";
  }
}

template <typename Domain, typename PolyOrEvals,
          typename RetPoly = std::conditional_t<
              std::is_same_v<PolyOrEvals, typename Domain::Evals>,
              typename Domain::DensePoly, typename Domain::Evals>>
void Run(const FFTConfig& config) {
  std::string_view name;
  if (config.run_ifft()) {
    name = "IFFT Benchmark";
  } else {
    name = "FFT Benchmark";
  }
  SimpleFFTBenchmarkReporter reporter(name, config.exponents());
  for (const FFTConfig::Vendor vendor : config.vendors()) {
    reporter.AddVendor(FFTConfig::VendorToString(vendor));
  }

  std::vector<size_t> degrees = config.GetDegrees();

  std::cout << "Generating evaluation domain and random polys..." << std::endl;
  std::vector<std::unique_ptr<Domain>> domains = base::Map(
      degrees, [](size_t degree) { return Domain::Create(degree + 1); });
  std::vector<PolyOrEvals> polys = base::Map(
      degrees, [](size_t degree) { return PolyOrEvals::Random(degree); });
  std::cout << "Generation completed" << std::endl;

  FFTRunner<Domain, PolyOrEvals> runner(&reporter);
  runner.SetInputs(&polys, std::move(domains));

  std::vector<RetPoly> results;
  if constexpr (std::is_same_v<PolyOrEvals, typename Domain::Evals>) {
    runner.Run(tachyon_bn254_univariate_evaluation_domain_ifft_inplace, degrees,
               &results);
    for (const FFTConfig::Vendor vendor : config.vendors()) {
      std::vector<RetPoly> results_vendor;
      if (vendor == FFTConfig::Vendor::kArkworks) {
        runner.RunExternal(run_ifft_arkworks, config.exponents(),
                           &results_vendor);
      } else if (vendor == FFTConfig::Vendor::kBellman) {
        runner.RunExternal(run_ifft_bellman, config.exponents(),
                           &results_vendor);
      } else if (vendor == FFTConfig::Vendor::kHalo2) {
        runner.RunExternal(run_ifft_halo2, config.exponents(), &results_vendor);
      }
      CheckResults(config.check_results(), results, results_vendor);
    }
    // NOLINTNEXTLINE(readability/braces)
  } else if constexpr (std::is_same_v<PolyOrEvals,
                                      typename Domain::DensePoly>) {
    runner.Run(tachyon_bn254_univariate_evaluation_domain_fft_inplace, degrees,
               &results);
    for (const FFTConfig::Vendor vendor : config.vendors()) {
      std::vector<RetPoly> results_vendor;
      if (vendor == FFTConfig::Vendor::kArkworks) {
        runner.RunExternal(run_fft_arkworks, config.exponents(),
                           &results_vendor);
      } else if (vendor == FFTConfig::Vendor::kBellman) {
        runner.RunExternal(run_fft_bellman, config.exponents(),
                           &results_vendor);
      } else if (vendor == FFTConfig::Vendor::kHalo2) {
        runner.RunExternal(run_fft_halo2, config.exponents(), &results_vendor);
      }
      CheckResults(config.check_results(), results, results_vendor);
    }
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
  math::halo2::OverrideSubgroupGenerator();

  FFTConfig config;
  FFTConfig::Options options;
  options.include_vendors = true;
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

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
