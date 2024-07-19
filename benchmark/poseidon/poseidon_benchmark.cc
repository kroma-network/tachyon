#include <iostream>

// clang-format off
#include "benchmark/poseidon/poseidon_config.h"
#include "benchmark/poseidon/poseidon_benchmark_runner.h"
#include "benchmark/poseidon/simple_poseidon_benchmark_reporter.h"
// clang-format on
#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon {

using namespace crypto;

using Field = math::bn254::Fr;

extern "C" tachyon_bn254_fr* run_poseidon_arkworks(uint64_t* duration);

int RealMain(int argc, char** argv) {
  tachyon::PoseidonConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  Field::Init();
  SimplePoseidonBenchmarkReporter reporter("Poseidon Benchmark",
                                           config.repeating_num());
  reporter.AddVendor("arkworks");
  PoseidonBenchmarkRunner<Field> runner(&reporter, &config);

  Field result = runner.Run();
  Field result_arkworks = runner.RunExternal(run_poseidon_arkworks);

  if (config.check_results()) {
    CHECK_EQ(result, result_arkworks) << "Result not matched";
  }

  reporter.AddAverageToLastRow();
  reporter.Show();

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
