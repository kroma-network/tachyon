#include <iostream>

// clang-format off
#include "benchmark/msm/msm_config.h"
#include "benchmark/msm/msm_runner.h"
#include "benchmark/msm/simple_msm_benchmark_reporter.h"
// clang-format on
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/msm.h"

namespace tachyon {

using namespace math;

extern "C" tachyon_bn254_g1_jacobian* run_msm_arkworks(
    const tachyon_bn254_g1_affine* bases, const tachyon_bn254_fr* scalars,
    size_t size, uint64_t* duration_in_us);

extern "C" tachyon_bn254_g1_jacobian* run_msm_bellman(
    const tachyon_bn254_g1_affine* bases, const tachyon_bn254_fr* scalars,
    size_t size, uint64_t* duration_in_us);

extern "C" tachyon_bn254_g1_projective* run_msm_halo2(
    const tachyon_bn254_g1_affine* bases, const tachyon_bn254_fr* scalars,
    size_t size, uint64_t* duration_in_us);

tachyon_bn254_g1_jacobian* run_msm_halo2_adapter(
    const tachyon_bn254_g1_affine* bases, const tachyon_bn254_fr* scalars,
    size_t size, uint64_t* duration_in_us) {
  std::unique_ptr<tachyon_bn254_g1_projective> projective(
      run_msm_halo2(bases, scalars, size, duration_in_us));
  return c::base::c_cast(new bn254::G1JacobianPoint(
      c::base::native_cast(projective.get())->ToJacobian()));
}

int RealMain(int argc, char** argv) {
  MSMConfig config;
  MSMConfig::Options options;
  options.include_vendors = true;
  if (!config.Parse(argc, argv, options)) {
    return 1;
  }

  SimpleMSMBenchmarkReporter reporter("MSM Benchmark", config.exponents());
  reporter.AddVendor("tachyon");
  for (const MSMConfig::Vendor vendor : config.vendors()) {
    reporter.AddVendor(MSMConfig::VendorToString(vendor));
  }

  std::vector<uint64_t> point_nums = config.GetPointNums();

  tachyon_bn254_g1_init();
  tachyon_bn254_g1_msm_ptr msm =
      tachyon_bn254_g1_create_msm(config.exponents().back());

  std::cout << "Generating random points..." << std::endl;
  uint64_t max_point_num = point_nums.back();
  VariableBaseMSMTestSet<bn254::G1AffinePoint> test_set;
  CHECK(config.GenerateTestSet(max_point_num, &test_set));
  std::cout << "Generation completed" << std::endl;

  MSMRunner<bn254::G1AffinePoint> runner(&reporter);
  runner.SetInputs(&test_set.bases, &test_set.scalars);
  std::vector<bn254::G1JacobianPoint> results;
  runner.Run(tachyon_bn254_g1_affine_msm, msm, point_nums, &results);
  for (const MSMConfig::Vendor vendor : config.vendors()) {
    std::vector<bn254::G1JacobianPoint> results_vendor;
    switch (vendor) {
      case MSMConfig::Vendor::kArkworks:
        runner.RunExternal(run_msm_arkworks, point_nums, &results_vendor);
        break;
      case MSMConfig::Vendor::kBellman:
        runner.RunExternal(run_msm_bellman, point_nums, &results_vendor);
        break;
      case MSMConfig::Vendor::kHalo2:
        runner.RunExternal(run_msm_halo2_adapter, point_nums, &results_vendor);
        break;
    }

    if (config.check_results()) {
      CHECK(results == results_vendor) << "Result not matched";
    }
  }

  reporter.Show();

  tachyon_bn254_g1_destroy_msm(msm);

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
