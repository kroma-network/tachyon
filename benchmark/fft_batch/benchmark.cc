#include <iostream>

// clang-format off
#include "benchmark/fft_batch/benchmark_reporter.h"
#include "benchmark/fft_batch/benchmark_runner.h"
#include "benchmark/fft_batch/config.h"
// clang-format on
#include "tachyon/base/logging.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::benchmark::fft_batch {

extern "C" void* run_fft_batch_plonky3_baby_bear(uint64_t* duration,
                                                 math::BabyBear* data,
                                                 size_t n_log,
                                                 size_t batch_size);

template <typename F, typename Fn>
void Run(BenchmarkReporter& reporter, const tachyon::Config& config,
         Fn plonky3_fn) {
  F::Init();

  BenchmarkRunner<F> runner(&reporter, &config);
  runner.Prepare();

  math::RowMajorMatrix<F> result = runner.Run();
  math::RowMajorMatrix<F> result_vendor;
  for (const Config::Vendor vendor : config.vendors()) {
    switch (vendor) {
      case Config::Vendor::kPlonky3:
        result_vendor = runner.RunExternal(plonky3_fn);
        break;
    }
  }
  if (config.check_results()) {
    NOTIMPLEMENTED() << "Check results not implemented";
    CHECK_EQ(result, result_vendor) << "Result not matched";
  }
}

int RealMain(int argc, char** argv) {
  Config config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  BenchmarkReporter reporter("FFTBatch Benchmark (" +
                                 std::to_string(config.degree()) + ", " +
                                 std::to_string(config.batch_size()) + ")",
                             config.repeating_num());

  for (const Config::Vendor vendor : config.vendors()) {
    reporter.AddVendor(Config::VendorToString(vendor));
  }

  switch (config.prime_field()) {
    case Config::PrimeField::kBabyBear: {
      Run<math::BabyBear>(reporter, config, run_fft_batch_plonky3_baby_bear);
      break;
    }
  }

  reporter.AddAverageToLastRow();
  reporter.Show();

  return 0;
}

}  // namespace tachyon::benchmark::fft_batch

int main(int argc, char** argv) {
  return tachyon::benchmark::fft_batch::RealMain(argc, argv);
}
