#include <iostream>

// clang-format off
#include "benchmark/poseidon/poseidon_config.h"
#include "benchmark/poseidon/poseidon_benchmark_runner.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::benchmark {

using Field = math::bn254::Fr;

extern "C" tachyon_bn254_fr* run_poseidon_arkworks(uint64_t* duration);

int RealMain(int argc, char** argv) {
  PoseidonConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  Field::Init();
  SimpleReporter reporter("Poseidon Benchmark");
  PoseidonBenchmarkRunner<Field> runner(reporter, config);

  reporter.set_x_label("Trial number");
  reporter.set_column_labels(
      base::CreateVector(config.repeating_num(),
                         [](size_t i) { return base::NumberToString(i); }));

  Field result = runner.Run();
  Field result_arkworks =
      runner.RunExternal(Vendor::Arkworks(), run_poseidon_arkworks);

  if (config.check_results()) {
    CHECK_EQ(result, result_arkworks) << "Result not matched";
  }

  reporter.AddAverageAsLastColumn();
  reporter.Show();

  return 0;
}

}  // namespace tachyon::benchmark

int main(int argc, char** argv) {
  return tachyon::benchmark::RealMain(argc, argv);
}
