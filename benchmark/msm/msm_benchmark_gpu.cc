#include <iostream>
#include <vector>
#if TACHYON_CUDA

// clang-format off
#include "benchmark/msm/msm_config.h"
#include "benchmark/msm/msm_runner.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/profiler.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/msm.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/msm_gpu.h"
#include "tachyon/math/elliptic_curves/msm/test/variable_base_msm_test_set.h"

namespace tachyon::benchmark {

using namespace math;

int RealMain(int argc, char** argv) {
  base::FilePath tmp_file;
  CHECK(base::GetTempDir(&tmp_file));
  tmp_file = tmp_file.Append("msm_benchmark_gpu.perfetto-trace");
  base::Profiler profiler({tmp_file});

  profiler.Init();
  profiler.Start();

  MSMConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  SimpleReporter reporter;
  reporter.set_title("MSM Benchmark");
  reporter.set_x_label("Degree (2ˣ)");
  reporter.set_column_labels(base::Map(
      config.exponents(),
      [](uint32_t exponent) { return base::NumberToString(exponent); }));

  std::vector<size_t> point_nums = config.GetPointNums();

  tachyon_bn254_g1_init();
  tachyon_bn254_g1_msm_ptr msm =
      tachyon_bn254_g1_create_msm(config.exponents().back());

  std::cout << "Generating random points..." << std::endl;
  size_t max_point_num = point_nums.back();
  VariableBaseMSMTestSet<bn254::G1AffinePoint> test_set;
  CHECK(config.GenerateTestSet(max_point_num, &test_set));
  std::cout << "Generation completed" << std::endl;

  MSMRunner<bn254::G1AffinePoint> runner(reporter);
  runner.SetInputs(test_set.bases, test_set.scalars);

  std::vector<bn254::G1JacobianPoint> results_cpu;
  runner.Run(Vendor::TachyonCPU(), tachyon_bn254_g1_affine_msm, msm, point_nums,
             results_cpu);
  tachyon_bn254_g1_destroy_msm(msm);

  tachyon_bn254_g1_msm_gpu_ptr msm_gpu =
      tachyon_bn254_g1_create_msm_gpu(config.exponents().back());
  std::vector<bn254::G1JacobianPoint> results_gpu;
  runner.Run(Vendor::TachyonGPU(), tachyon_bn254_g1_affine_msm_gpu, msm_gpu,
             point_nums, results_gpu);
  tachyon_bn254_g1_destroy_msm_gpu(msm_gpu);

  if (config.check_results()) {
    CHECK(results_cpu == results_gpu) << "Results do not match";
  }

  reporter.Show();

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
