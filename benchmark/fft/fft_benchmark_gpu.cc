#if TACHYON_CUDA
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

  SimpleReporter reporter;
  reporter.set_title(name);
  reporter.set_x_label("Degree (2ˣ)");
  reporter.set_column_labels(base::Map(
      config.exponents(),
      [](uint32_t exponent) { return base::NumberToString(exponent); }));

  reporter.AddVendor(Vendor::TachyonCPU());
  reporter.AddVendor(Vendor::TachyonGPU());

  std::vector<size_t> degrees = config.GetDegrees();

  FFTRunner<Domain, PolyOrEvals> runner(reporter);

  for (size_t degree : degrees) {
    PolyOrEvals input = PolyOrEvals::Random(degree);
    std::unique_ptr<Domain> domain = Domain::Create(degree + 1);
    bool kShouldRecord = true;

    IcicleNTTHolder<F> icicle_ntt_holder = IcicleNTTHolder<F>::Create();
    CHECK(icicle_ntt_holder->Init(domain->group_gen()));

    RetPoly cpu_result, gpu_result;
    if constexpr (std::is_same_v<PolyOrEvals, typename Domain::Evals>) {
      runner.Run(Vendor::TachyonCPU(),
                 tachyon_bn254_univariate_evaluation_domain_ifft_inplace,
                 domain.get(), input, kShouldRecord, cpu_result);

      domain->set_icicle(&icicle_ntt_holder);

      runner.Run(Vendor::TachyonGPU(),
                 tachyon_bn254_univariate_evaluation_domain_ifft_inplace,
                 domain.get(), input, kShouldRecord, gpu_result);
      // NOLINTNEXTLINE(readability/braces)
    } else if constexpr (std::is_same_v<PolyOrEvals,
                                        typename Domain::DensePoly>) {
      runner.Run(Vendor::TachyonCPU(),
                 tachyon_bn254_univariate_evaluation_domain_fft_inplace,
                 domain.get(), input, kShouldRecord, cpu_result);

      domain->set_icicle(&icicle_ntt_holder);

      runner.Run(Vendor::TachyonGPU(),
                 tachyon_bn254_univariate_evaluation_domain_fft_inplace,
                 domain.get(), input, kShouldRecord, gpu_result);
    }
    if (config.check_results()) {
      CHECK_EQ(cpu_result, gpu_result) << "Results not matched";
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
  tmp_file = tmp_file.Append("fft_benchmark_gpu.perfetto-trace");
  base::Profiler profiler({tmp_file});

  profiler.Init();
  profiler.Start();

  Field::Init();

  FFTConfig config;
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
#else
#include "tachyon/base/console/iostream.h"

int main(int argc, char **argv) {
  tachyon_cerr << "please build with --config cuda" << std::endl;
  return 1;
}
#endif  // TACHYON_CUDA
