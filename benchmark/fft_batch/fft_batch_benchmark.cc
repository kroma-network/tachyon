#include <iostream>

#include "absl/strings/substitute.h"

// clang-format off
#include "benchmark/fft_batch/fft_batch_config.h"
#include "benchmark/fft_batch/fft_batch_runner.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/profiler.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/math/finite_fields/packed_field_traits_forward.h"

namespace tachyon::benchmark {

extern "C" tachyon_baby_bear* run_fft_batch_plonky3_baby_bear(
    const tachyon_baby_bear* data, uint32_t n_log, size_t batch_size,
    uint64_t* duration);

extern "C" tachyon_baby_bear* run_coset_lde_batch_plonky3_baby_bear(
    const tachyon_baby_bear* data, uint32_t n_log, size_t batch_size,
    uint64_t* duration);

template <typename F>
void CheckResults(bool check_results,
                  const math::RowMajorMatrix<F>& tachyon_result,
                  const math::RowMajorMatrix<F>& vendor_result) {
  if (check_results) {
    CHECK_EQ(tachyon_result, vendor_result) << "Results not matched";
  }
}

template <typename F>
void Run(const FFTBatchConfig& config) {
  using Domain = math::Radix2EvaluationDomain<F, SIZE_MAX - 1>;
  using PackedPrimeField =
      // NOLINTNEXTLINE(whitespace/operators)
      std::conditional_t<F::Config::kModulusBits <= 32,
                         typename math::PackedFieldTraits<F>::PackedField, F>;

  PackedPrimeField::Init();

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

  FFTBatchRunner<Domain> runner(reporter, config);

  reporter.AddVendor(Vendor::Tachyon());
  for (Vendor vendor : config.vendors()) {
    reporter.AddVendor(vendor);
  }

  for (size_t degree : degrees) {
    math::RowMajorMatrix<F> input =
        math::RowMajorMatrix<F>::Random(degree, config.batch_size());

    math::RowMajorMatrix<F> tachyon_result =
        runner.Run(Vendor::Tachyon(), config.run_coset_lde(), input);
    for (Vendor vendor : config.vendors()) {
      math::RowMajorMatrix<F> vendor_result;
      if (vendor.value() == Vendor::kPlonky3) {
        if (config.run_coset_lde()) {
          vendor_result = runner.RunExternal(
              vendor, run_coset_lde_batch_plonky3_baby_bear, input);
        } else {
          vendor_result = runner.RunExternal(
              vendor, run_fft_batch_plonky3_baby_bear, input);
        }
        CheckResults(config.check_results(), tachyon_result, vendor_result);
      } else {
        NOTREACHED();
      }
    }
  }

  reporter.Show();
}

int RealMain(int argc, char** argv) {
  base::FilePath tmp_file;
  CHECK(base::GetTempDir(&tmp_file));
  tmp_file = tmp_file.Append("fft_batch_benchmark.perfetto-trace");
  base::Profiler profiler({tmp_file});

  profiler.Init();
  profiler.Start();

  FFTBatchConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  if (config.prime_field().value() == FieldType::kBabyBear) {
    Run<math::BabyBear>(config);
  } else {
    NOTREACHED();
  }
  return 0;
}

}  // namespace tachyon::benchmark

int main(int argc, char** argv) {
  return tachyon::benchmark::RealMain(argc, argv);
}
