#if TACHYON_CUDA
#include <iostream>

// clang-format off
#include "benchmark/ec/ec_util.h"
#include "benchmark/ec/simple_ec_benchmark_reporter.h"
#include "benchmark/msm/msm_config.h"
// clang-format on
#include "tachyon/base/time/time_interval.h"
#include "tachyon/c/math/elliptic_curves/msm/msm.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_gpu.h"

namespace tachyon {

using namespace math;

int RealMain(int argc, char** argv) {
  ECConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  SimpleECBenchmarkReporter reporter(config.degrees());

  std::vector<uint64_t> point_nums = config.GetPointNums();

  std::cout << "Generating random points..." << std::endl;
  uint64_t max_point_num = point_nums.back();
  std::vector<bn254::G1AffinePoint> bases =
      CreateRandomBn254Points(max_point_num);
  std::vector<bn254::Fr> scalars = CreateRandomBn254Scalars(max_point_num);
  std::cout << "Generation completed" << std::endl;

  tachyon_init_msm(config.degrees().back());

  base::TimeInterval interval(base::TimeTicks::Now());
  std::vector<bn254::G1JacobianPoint> results_cpu;
  for (uint64_t point_num : point_nums) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    ret.reset(tachyon_bn254_g1_affine_msm(
        reinterpret_cast<const tachyon_bn254_g1_affine*>(bases.data()),
        point_num, reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
        point_num));
    results_cpu.push_back(
        *reinterpret_cast<bn254::G1JacobianPoint*>(ret.get()));
    auto duration = interval.GetTimeDelta();
    std::cout << "calculate:" << duration.InSecondsF() << std::endl;
    reporter.AddResult(duration.InSecondsF());
  }

  tachyon_release_msm();

  tachyon_init_msm_gpu(config.degrees().back());

  interval.Reset();
  std::vector<bn254::G1JacobianPoint> results_gpu;
  for (uint64_t point_num : point_nums) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    ret.reset(tachyon_msm_g1_affine_gpu(
        reinterpret_cast<const tachyon_bn254_g1_affine*>(bases.data()),
        point_num, reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
        point_num));
    results_gpu.push_back(
        *reinterpret_cast<bn254::G1JacobianPoint*>(ret.get()));
    reporter.AddResult(interval.GetTimeDelta().InSecondsF());
  }

  tachyon_release_msm_gpu();

  CHECK(results_cpu == results_gpu) << "Result not matched";

  reporter.Show();

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
#else
#include "tachyon/base/console/iostream.h"

int main(int argc, char **argv) {
  tachyon_cerr << "please build with --config cuda" << std::endl;
  return 1;
}
#endif  // TACHYON_CUDA
