#include <stddef.h>
#include <stdint.h>

#include <iostream>

// clang-format off
#include "benchmark/fft/fft_config.h"
#include "benchmark/fft/fft_runner.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/console/iostream.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/elliptic_curves/bn/bn254/halo2/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::benchmark {

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

  SimpleReporter reporter;
  reporter.set_title(name);
  reporter.set_x_label("Degree (2ˣ)");
  reporter.set_column_labels(base::Map(
      config.exponents(),
      [](uint32_t exponent) { return base::NumberToString(exponent); }));

  std::vector<size_t> degrees = config.GetDegrees();

  std::cout << "Generating evaluation domain and random polys..." << std::endl;
  std::vector<std::unique_ptr<Domain>> domains = base::Map(
      degrees, [](size_t degree) { return Domain::Create(degree + 1); });
  std::vector<std::unique_ptr<Domain>> halo2_domains;
  for (const Vendor vendor : config.vendors()) {
    if (vendor.value() == Vendor::kBellman ||
        vendor.value() == Vendor::kScrollHalo2) {
      math::halo2::ScopedSubgroupGeneratorOverrider scoped_overrider;
      halo2_domains = base::Map(
          degrees, [](size_t degree) { return Domain::Create(degree + 1); });
      break;
    }
  }
  std::vector<PolyOrEvals> polys = base::Map(
      degrees, [](size_t degree) { return PolyOrEvals::Random(degree); });
  std::cout << "Generation completed" << std::endl;

  FFTRunner<Domain, PolyOrEvals> runner(reporter);
  runner.set_polys(polys);

  std::vector<RetPoly> results;
  std::vector<RetPoly> halo2_results;
  if constexpr (std::is_same_v<PolyOrEvals, typename Domain::Evals>) {
    runner.set_domains(absl::MakeSpan(domains));
    runner.Run(Vendor::TachyonCPU(),
               tachyon_bn254_univariate_evaluation_domain_ifft_inplace, degrees,
               results, true);
    if (!halo2_domains.empty()) {
      runner.set_domains(absl::MakeSpan(halo2_domains));
      runner.Run(Vendor::TachyonCPU(),
                 tachyon_bn254_univariate_evaluation_domain_ifft_inplace,
                 degrees, halo2_results, false);
    }
    for (const Vendor vendor : config.vendors()) {
      std::vector<RetPoly> results_vendor;
      if (vendor.value() == Vendor::kArkworks) {
        runner.set_domains(absl::MakeSpan(domains));
        runner.RunExternal(vendor, run_ifft_arkworks, config.exponents(),
                           results_vendor);
        CheckResults(config.check_results(), results, results_vendor);
      } else if (vendor.value() == Vendor::kBellman) {
        runner.set_domains(absl::MakeSpan(halo2_domains));
        runner.RunExternal(vendor, run_ifft_bellman, config.exponents(),
                           results_vendor);
        CheckResults(config.check_results(), halo2_results, results_vendor);
      } else if (vendor.value() == Vendor::kScrollHalo2) {
        runner.set_domains(absl::MakeSpan(halo2_domains));
        runner.RunExternal(vendor, run_ifft_halo2, config.exponents(),
                           results_vendor);
        CheckResults(config.check_results(), halo2_results, results_vendor);
      } else {
        tachyon_cerr << "Unsupported vendor " << vendor.ToString() << std::endl;
      }
    }
    // NOLINTNEXTLINE(readability/braces)
  } else if constexpr (std::is_same_v<PolyOrEvals,
                                      typename Domain::DensePoly>) {
    runner.set_domains(absl::MakeSpan(domains));
    runner.Run(Vendor::TachyonCPU(),
               tachyon_bn254_univariate_evaluation_domain_fft_inplace, degrees,
               results, true);
    if (!halo2_domains.empty()) {
      runner.set_domains(absl::MakeSpan(halo2_domains));
      runner.Run(Vendor::TachyonCPU(),
                 tachyon_bn254_univariate_evaluation_domain_fft_inplace,
                 degrees, halo2_results, false);
    }
    for (const Vendor vendor : config.vendors()) {
      std::vector<RetPoly> results_vendor;
      if (vendor.value() == Vendor::kArkworks) {
        runner.set_domains(absl::MakeSpan(domains));
        runner.RunExternal(vendor, run_fft_arkworks, config.exponents(),
                           results_vendor);
        CheckResults(config.check_results(), results, results_vendor);
      } else if (vendor.value() == Vendor::kBellman) {
        runner.set_domains(absl::MakeSpan(halo2_domains));
        runner.RunExternal(vendor, run_fft_bellman, config.exponents(),
                           results_vendor);
        CheckResults(config.check_results(), halo2_results, results_vendor);
      } else if (vendor.value() == Vendor::kScrollHalo2) {
        runner.set_domains(absl::MakeSpan(halo2_domains));
        runner.RunExternal(vendor, run_fft_halo2, config.exponents(),
                           results_vendor);
        CheckResults(config.check_results(), halo2_results, results_vendor);
      } else {
        tachyon_cerr << "Unsupported vendor " << vendor.ToString() << std::endl;
      }
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

}  // namespace tachyon::benchmark

int main(int argc, char** argv) {
  return tachyon::benchmark::RealMain(argc, argv);
}
