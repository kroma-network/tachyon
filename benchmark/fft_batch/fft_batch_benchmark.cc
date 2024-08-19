#include <iostream>

#include "absl/strings/substitute.h"

// clang-format off
#include "benchmark/fft_batch/fft_batch_config.h"
#include "benchmark/fft_batch/fft_batch_runner.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/console/iostream.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"

namespace tachyon::benchmark {

extern "C" tachyon_baby_bear* run_fft_batch_plonky3_baby_bear(
    const tachyon_baby_bear* data, uint32_t n_log, size_t batch_size,
    uint64_t* duration);

extern "C" tachyon_baby_bear* run_coset_lde_batch_plonky3_baby_bear(
    const tachyon_baby_bear* data, uint32_t n_log, size_t batch_size,
    uint64_t* duration);

template <typename F>
void CheckResults(bool check_results,
                  const std::vector<math::RowMajorMatrix<F>>& results,
                  const std::vector<math::RowMajorMatrix<F>>& results_vendor) {
  if (check_results) {
    CHECK(results == results_vendor) << "Results not matched";
  }
}

template <typename F>
int Run(const FFTBatchConfig& config) {
  using Domain = math::Radix2EvaluationDomain<F, SIZE_MAX - 1>;

  F::Init();

  std::string name;
  if (config.run_coset_lde()) {
    name = absl::Substitute("CosetLDEBatch Benchmark (Batch Size: $0)",
                            config.batch_size());
  } else {
    name = absl::Substitute("FFTBatch Benchmark (Batch Size: $0)",
                            config.batch_size());
  }

  SimpleReporter reporter;
  reporter.set_title(name);
  reporter.set_x_label("Degree (2ˣ)");
  reporter.set_column_labels(base::Map(
      config.exponents(),
      [](uint32_t exponent) { return base::NumberToString(exponent); }));

  std::vector<size_t> degrees = config.GetDegrees();

  std::cout << "Generating evaluation domain and random matrices..."
            << std::endl;
  std::vector<std::unique_ptr<Domain>> domains =
      base::Map(degrees, [](size_t degree) { return Domain::Create(degree); });
  std::vector<math::RowMajorMatrix<F>> inputs =
      base::Map(degrees, [&config](size_t degree) {
        math::RowMajorMatrix<F> matrix =
            math::RowMajorMatrix<F>::Random(degree, config.batch_size());
        return matrix;
      });

  FFTBatchRunner<Domain> runner(reporter, config);
  runner.set_domains(absl::MakeSpan(domains));
  runner.set_inputs(absl::MakeSpan(inputs));

  std::vector<math::RowMajorMatrix<F>> results;
  runner.Run(Vendor::TachyonCPU(), results, config.run_coset_lde());
  for (Vendor vendor : config.vendors()) {
    std::vector<math::RowMajorMatrix<F>> results_vendor;
    if (vendor.value() == Vendor::kPlonky3) {
      if (config.run_coset_lde()) {
        runner.RunExternal(vendor, run_coset_lde_batch_plonky3_baby_bear,
                           results_vendor);
      } else {
        runner.RunExternal(vendor, run_fft_batch_plonky3_baby_bear,
                           results_vendor);
      }
      CheckResults(config.check_results(), results, results_vendor);
    } else {
      tachyon_cerr << "Unsupported vendor\n";
      return 1;
    }
  }

  reporter.Show();
  return 0;
}

int RealMain(int argc, char** argv) {
  FFTBatchConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  if (config.prime_field().value() == FieldType::kBabyBear) {
    return Run<math::BabyBear>(config);
  } else {
    tachyon_cerr << "Unsupported prime field\n";
    return 1;
  }
}

}  // namespace tachyon::benchmark

int main(int argc, char** argv) {
  return tachyon::benchmark::RealMain(argc, argv);
}
