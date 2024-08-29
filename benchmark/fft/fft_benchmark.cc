#include <stddef.h>
#include <stdint.h>

#include <iostream>

// clang-format off
#include "benchmark/fft/fft_config.h"
#include "benchmark/fft/fft_runner.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/profiler.h"
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
void CheckResult(bool check_result, const PolyOrEvals& tachyon_result,
                 const PolyOrEvals& vendor_result) {
  if (check_result) {
    CHECK_EQ(tachyon_result, vendor_result) << "Results not matched";
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

  reporter.AddVendor(Vendor::Tachyon());
  for (const Vendor vendor : config.vendors()) {
    reporter.AddVendor(vendor);
  }

  FFTRunner<Domain, PolyOrEvals> runner(reporter);

  std::vector<size_t> degrees = config.GetDegrees();

  bool need_halo2_results =
      base::Contains(config.vendors(), Vendor::Bellman()) ||
      base::Contains(config.vendors(), Vendor::ScrollHalo2());
  for (size_t degree : degrees) {
    PolyOrEvals input = PolyOrEvals::Random(degree);
    std::unique_ptr<Domain> domain = Domain::Create(degree + 1);
    std::unique_ptr<Domain> halo2_domain;
    if (need_halo2_results) {
      math::halo2::ScopedSubgroupGeneratorOverrider scoped_overrider;
      halo2_domain = Domain::Create(degree + 1);
    }

    if constexpr (std::is_same_v<PolyOrEvals, typename Domain::Evals>) {
      RetPoly tachyon_result, tachyon_halo2_result;
      runner.Run(Vendor::Tachyon(),
                 tachyon_bn254_univariate_evaluation_domain_ifft_inplace,
                 domain.get(), input, true, tachyon_result);
      if (halo2_domain) {
        runner.Run(Vendor::Tachyon(),
                   tachyon_bn254_univariate_evaluation_domain_ifft_inplace,
                   halo2_domain.get(), input, false, tachyon_halo2_result);
      }

      for (const Vendor vendor : config.vendors()) {
        if (vendor.value() == Vendor::kArkworks) {
          RetPoly vendor_result;
          runner.RunExternal(vendor, run_ifft_arkworks, domain.get(), input,
                             vendor_result);
          CheckResult(config.check_results(), tachyon_result, vendor_result);
        } else if (vendor.value() == Vendor::kBellman) {
          RetPoly vendor_result;
          runner.RunExternal(vendor, run_ifft_bellman, halo2_domain.get(),
                             input, vendor_result);
          CheckResult(config.check_results(), tachyon_halo2_result,
                      vendor_result);
        } else if (vendor.value() == Vendor::kScrollHalo2) {
          RetPoly vendor_result;
          runner.RunExternal(vendor, run_ifft_halo2, halo2_domain.get(), input,
                             vendor_result);
          CheckResult(config.check_results(), tachyon_halo2_result,
                      vendor_result);
        } else {
          NOTREACHED();
        }
      }
      // NOLINTNEXTLINE(readability/braces)
    } else if constexpr (std::is_same_v<PolyOrEvals,
                                        typename Domain::DensePoly>) {
      RetPoly tachyon_result, tachyon_halo2_result;
      runner.Run(Vendor::Tachyon(),
                 tachyon_bn254_univariate_evaluation_domain_fft_inplace,
                 domain.get(), input, true, tachyon_result);
      if (halo2_domain) {
        runner.Run(Vendor::Tachyon(),
                   tachyon_bn254_univariate_evaluation_domain_fft_inplace,
                   halo2_domain.get(), input, false, tachyon_halo2_result);
      }

      for (const Vendor vendor : config.vendors()) {
        if (vendor.value() == Vendor::kArkworks) {
          RetPoly vendor_result;
          runner.RunExternal(vendor, run_fft_arkworks, domain.get(), input,
                             vendor_result);
          CheckResult(config.check_results(), tachyon_result, vendor_result);
        } else if (vendor.value() == Vendor::kBellman) {
          RetPoly vendor_result;
          runner.RunExternal(vendor, run_fft_bellman, halo2_domain.get(), input,
                             vendor_result);
          CheckResult(config.check_results(), tachyon_halo2_result,
                      vendor_result);
        } else if (vendor.value() == Vendor::kScrollHalo2) {
          RetPoly vendor_result;
          runner.RunExternal(vendor, run_fft_halo2, halo2_domain.get(), input,
                             vendor_result);
          CheckResult(config.check_results(), tachyon_halo2_result,
                      vendor_result);
        } else {
          NOTREACHED();
        }
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

  base::FilePath tmp_file;
  CHECK(base::GetTempDir(&tmp_file));
  tmp_file = tmp_file.Append("fft_benchmark.perfetto-trace");
  base::Profiler profiler({tmp_file});

  profiler.Init();
  profiler.Start();

  Field::Init();

  FFTConfig::Options options;
  options.include_vendors = true;
  FFTConfig config(options);
  if (!config.Parse(argc, argv)) {
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
