#include <iostream>

// clang-format off
#include "benchmark/poseidon/simple_poseidon_benchmark_reporter.h"
#include "benchmark/poseidon2/poseidon2_benchmark_runner.h"
#include "benchmark/poseidon2/poseidon2_config.h"
// clang-format on
#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_config.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/elliptic_curves/bn/bn254/poseidon2.h"

namespace tachyon {

using namespace crypto;

using Field = math::bn254::Fr;

extern "C" tachyon_bn254_fr* run_poseidon2_horizen_bn254_fr(uint64_t* duration);
extern "C" tachyon_bn254_fr* run_poseidon2_plonky3_bn254_fr(uint64_t* duration);

template <typename Field, typename Fn>
void Run(SimplePoseidonBenchmarkReporter& reporter,
         const tachyon::Poseidon2Config& config, Fn horizen_fn, Fn plonky3_fn) {
  Field::Init();

  Poseidon2BenchmarkRunner<Field> runner(&reporter, &config);

  crypto::Poseidon2Config<Field> poseidon2_config =
      crypto::Poseidon2Config<Field>::CreateCustom(
          2, 5, 8, 56, math::bn254::GetPoseidon2InternalDiagonalVector<3>());

  Field result = runner.Run(poseidon2_config);
  for (const tachyon::Poseidon2Config::Vendor vendor : config.vendors()) {
    Field result_vendor;
    switch (vendor) {
      case tachyon::Poseidon2Config::Vendor::kHorizen:
        result_vendor = runner.RunExternal(horizen_fn);
        break;
      case tachyon::Poseidon2Config::Vendor::kPlonky3:
        result_vendor = runner.RunExternal(plonky3_fn);
        break;
    }

    if (config.check_results()) {
      CHECK_EQ(result, result_vendor) << "Result not matched";
    }
  }
}

int RealMain(int argc, char** argv) {
  tachyon::Poseidon2Config config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  SimplePoseidonBenchmarkReporter reporter("Poseidon2 Benchmark",
                                           config.repeating_num());

  for (const tachyon::Poseidon2Config::Vendor vendor : config.vendors()) {
    reporter.AddVendor(tachyon::Poseidon2Config::VendorToString(vendor));
  }

  switch (config.prime_field()) {
    case tachyon::Poseidon2Config::PrimeField::kBn254Fr: {
      Run<math::bn254::Fr>(reporter, config, run_poseidon2_horizen_bn254_fr,
                           run_poseidon2_plonky3_bn254_fr);
      break;
    }
  }

  reporter.AddAverageToLastRow();
  reporter.Show();

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
