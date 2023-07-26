#if TACHYON_CUDA
#include <iostream>

#include "third_party/matplotlibcpp17/include/animation.h"
#include "third_party/matplotlibcpp17/include/pyplot.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/base/time/time_interval.h"
#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_cuda.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"

using namespace matplotlibcpp17;

namespace tachyon {

namespace {

std::vector<math::bn254::G1JacobianPoint> CreateRandomPoints(size_t nums) {
  std::vector<math::bn254::G1JacobianPoint> ret;
  ret.reserve(nums);
  for (size_t i = 0; i < nums; ++i) {
    ret.push_back(math::bn254::G1JacobianPoint::Random());
  }
  return ret;
}

// TODO(chokobole): Use openmp.
void TestDoubleOnCPU(const std::vector<math::bn254::G1JacobianPoint>& bases,
                     std::vector<math::bn254::G1JacobianPoint>& results,
                     uint64_t nums) {
  for (uint64_t i = 0; i < nums; ++i) {
    results[i] = bases[i].Double();
  }
}

cudaError_t LaunchDouble(const math::bn254::G1JacobianPointCuda* x,
                         math::bn254::G1JacobianPointCuda* y, uint64_t count) {
  math::kernels::Double<<<(count - 1) / 32 + 1, 32>>>(x, y, count);
  cudaError_t error = cudaGetLastError();
  GPU_LOG_IF_ERROR(ERROR, error);
  error = error ? error : cudaDeviceSynchronize();
  GPU_LOG_IF_ERROR(ERROR, error);
  return error;
}

void TestDoubleOnGPU(math::bn254::G1JacobianPointCuda* bases_cuda,
                     math::bn254::G1JacobianPointCuda* results_cuda,
                     const std::vector<math::bn254::G1JacobianPoint>& bases,
                     uint64_t nums) {
  for (uint64_t i = 0; i < nums; ++i) {
    bases_cuda[i] = math::bn254::G1JacobianPointCuda::FromMontgomery(
        bases[i].ToMontgomery());
  }

  LaunchDouble(bases_cuda, results_cuda, nums);
}

}  // namespace

int RealMain(int argc, char** argv) {
  std::vector<uint64_t> test_nums;
  base::FlagParser parser;
  parser.AddFlag<base::Flag<std::vector<uint64_t>>>(&test_nums)
      .set_short_name("-n")
      .set_required()
      .set_help("The number of points to test");
  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
  }

  math::bn254::G1AffinePointCuda::Curve::Init();
  math::bn254::G1AffinePoint::Curve::Init();

  base::ranges::sort(test_nums);
  std::vector<std::string> names;
  std::vector<double> results;
  names.reserve(test_nums.size() * 2);
  results.reserve(test_nums.size() * 2);
  for (uint64_t test_num : test_nums) {
    names.push_back(absl::Substitute("CPU/$0", test_num));
  }
  for (uint64_t test_num : test_nums) {
    names.push_back(absl::Substitute("CUDA/$0", test_num));
  }

  std::cout << "Generating random points..." << std::endl;
  uint64_t max_nums = test_nums.back();
  std::vector<math::bn254::G1JacobianPoint> bases_cpu =
      CreateRandomPoints(max_nums);
  std::vector<math::bn254::G1JacobianPoint> results_cpu;
  results_cpu.resize(max_nums);
  std::cout << "Generation completed" << std::endl;
  base::TimeInterval interval(base::TimeTicks::Now());
  for (uint64_t test_num : test_nums) {
    TestDoubleOnCPU(bases_cpu, results_cpu, test_num);
    results.push_back(interval.GetTimeDelta().InSecondsF());
  }

  GPU_SUCCESS(cudaDeviceReset());
  auto bases_cuda =
      device::gpu::MakeManagedUnique<math::bn254::G1JacobianPointCuda>(
          max_nums * sizeof(math::bn254::G1JacobianPointCuda));
  auto results_cuda =
      device::gpu::MakeManagedUnique<math::bn254::G1JacobianPointCuda>(
          max_nums * sizeof(math::bn254::G1JacobianPointCuda));

  interval.Start();
  for (uint64_t test_num : test_nums) {
    TestDoubleOnGPU(bases_cuda.get(), results_cuda.get(), bases_cpu, test_num);
    results.push_back(interval.GetTimeDelta().InSecondsF());
  }

#if defined(TACHYON_HAS_MATPLOTLIB)
  py::scoped_interpreter guard{};
  auto plt = pyplot::import();

  auto [fig, ax] = plt.subplots(
      Kwargs("layout"_a = "constrained", "figsize"_a = py::make_tuple(12, 6)));

  ax.set_title(Args("Benchmark results"));

  ax.bar(Args(names, results));
  plt.show();
#endif  // defined(TACHYON_HAS_MATPLOTLIB)

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
