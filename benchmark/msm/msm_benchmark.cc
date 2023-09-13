#include <iostream>

// clang-format off
#include "benchmark/msm/msm_config.h"
#include "benchmark/msm/msm_runner.h"
#include "benchmark/msm/simple_msm_benchmark_reporter.h"
// clang-format on
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/msm/msm.h"
#include "tachyon/math/elliptic_curves/test/random.h"

namespace tachyon {

using namespace math;

extern "C" tachyon_bn254_g1_jacobian* run_msm_arkworks(
    const tachyon_bn254_g1_affine* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len,
    uint64_t* duration_in_us);

extern "C" tachyon_bn254_g1_jacobian* run_msm_bellman(
    const tachyon_bn254_g1_affine* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len,
    uint64_t* duration_in_us);

int RealMain(int argc, char** argv) {
  MSMConfig config;
  if (!config.Parse(argc, argv, true)) {
    return 1;
  }

  SimpleMSMBenchmarkReporter reporter(config.degrees());
  for (const MSMConfig::Vendor vendor : config.vendors()) {
    reporter.AddVendor(MSMConfig::VendorToString(vendor));
  }

  std::vector<uint64_t> point_nums = config.GetPointNums();

  tachyon_init_msm(config.degrees().back());

  std::cout << "Generating random points..." << std::endl;
  uint64_t max_point_num = point_nums.back();
  std::vector<bn254::G1AffinePoint> bases =
      CreatePseudoRandomPoints<bn254::G1AffinePoint>(max_point_num);
  std::vector<bn254::Fr> scalars =
      base::CreateVector(max_point_num, []() { return bn254::Fr::Random(); });
  std::cout << "Generation completed" << std::endl;

  MSMRunner<bn254::G1AffinePoint> runner(&reporter);
  runner.SetInputs(&bases, &scalars);
  std::vector<bn254::G1JacobianPoint> results;
  runner.Run(tachyon_bn254_g1_affine_msm, point_nums, &results);
  for (const MSMConfig::Vendor vendor : config.vendors()) {
    std::vector<bn254::G1JacobianPoint> results_vendor;
    if (vendor == MSMConfig::Vendor::kArkworks) {
      runner.RunExternal(run_msm_arkworks, point_nums, &results_vendor);
    } else if (vendor == MSMConfig::Vendor::kBellman) {
      runner.RunExternal(run_msm_bellman, point_nums, &results_vendor);
    }

    if (config.check_results()) {
      CHECK(results == results_vendor) << "Result not matched";
    }
  }

  reporter.Show();

  tachyon_release_msm();

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
