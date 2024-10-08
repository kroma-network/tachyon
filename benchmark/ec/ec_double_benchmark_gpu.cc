#if TACHYON_CUDA || TACHYON_USE_ROCM

#include <iostream>

// clang-format off
#include "benchmark/simple_reporter.h"
#include "benchmark/ec/ec_config.h"
// clang-format on
#include "tachyon/base/profiler.h"
#include "tachyon/base/time/time_interval.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/elliptic_curves/test/random.h"
#include "tachyon/math/geometry/point_conversions.h"

namespace tachyon::benchmark {

using namespace device;
using namespace math;

namespace {

// TODO(chokobole): Use openmp.
void TestDoubleOnCPU(const std::vector<math::bn254::G1AffinePoint>& bases,
                     std::vector<math::bn254::G1JacobianPoint>& results,
                     size_t nums) {
  for (size_t i = 0; i < nums; ++i) {
    results[i] = bases[i].Double();
  }
}

gpuError_t LaunchDouble(const math::bn254::G1AffinePointGpu* x,
                        math::bn254::G1JacobianPointGpu* y, size_t count) {
  math::kernels::Double<<<(count - 1) / 32 + 1, 32>>>(x, y, count);
  gpuError_t error = LOG_IF_GPU_LAST_ERROR("Failed Double()");
  return error == gpuSuccess ? LOG_IF_GPU_ERROR(gpuDeviceSynchronize(),
                                                "Failed gpuDeviceSynchronize()")
                             : error;
}

void TestDoubleOnGPU(math::bn254::G1AffinePointGpu* bases_cuda,
                     math::bn254::G1JacobianPointGpu* results_cuda,
                     const std::vector<math::bn254::G1AffinePoint>& bases,
                     size_t nums) {
  for (size_t i = 0; i < nums; ++i) {
    bases_cuda[i] = ConvertPoint<math::bn254::G1AffinePointGpu>(bases[i]);
  }

  LaunchDouble(bases_cuda, results_cuda, nums);
}

}  // namespace

int RealMain(int argc, char** argv) {
  base::FilePath tmp_file;
  CHECK(base::GetTempDir(&tmp_file));
  tmp_file = tmp_file.Append("ec_double_benchmark_gpu.perfetto-trace");
  base::Profiler profiler({tmp_file});

  profiler.Init();
  profiler.Start();

  ECConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  math::bn254::G1Curve::Init();
  math::bn254::G1CurveGpu::Init();

  const std::vector<size_t>& point_nums = config.point_nums();

  SimpleReporter reporter;
  reporter.set_title("EC double benchmark");
  reporter.set_x_label("# of points");
  reporter.set_column_labels(base::Map(
      point_nums, [](size_t num) { return base::NumberToString(num); }));

  std::cout << "Generating random points..." << std::endl;
  size_t max_point_num = point_nums.back();
  std::vector<bn254::G1AffinePoint> bases =
      CreatePseudoRandomPoints<bn254::G1AffinePoint>(max_point_num);
  std::vector<bn254::Fr> scalars = base::CreateVectorParallel(
      max_point_num, []() { return bn254::Fr::Random(); });
  std::cout << "Generation completed" << std::endl;

  std::vector<math::bn254::G1JacobianPoint> results_cpu;
  results_cpu.resize(max_point_num);
  reporter.AddVendor(Vendor::TachyonCPU());
  base::TimeInterval interval(base::TimeTicks::Now());
  for (size_t i = 0; i < point_nums.size(); ++i) {
    TestDoubleOnCPU(bases, results_cpu, point_nums[i]);
    reporter.AddTime(Vendor::TachyonCPU(), interval.GetTimeDelta());
  }

  GPU_MUST_SUCCEED(gpuDeviceReset(), "Failed gpuDeviceReset()");
  auto bases_cuda =
      gpu::GpuMemory<math::bn254::G1AffinePointGpu>::MallocManaged(
          max_point_num);
  auto results_cuda =
      gpu::GpuMemory<math::bn254::G1JacobianPointGpu>::MallocManaged(
          max_point_num);

  reporter.AddVendor(Vendor::TachyonGPU());
  interval.Reset();
  for (size_t i = 0; i < point_nums.size(); ++i) {
    TestDoubleOnGPU(bases_cuda.data(), results_cuda.data(), bases,
                    point_nums[i]);
    reporter.AddTime(Vendor::TachyonGPU(), interval.GetTimeDelta());
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
  tachyon_cerr << "please build with --config cuda or --config rocm"
               << std::endl;
  return 1;
}
#endif  // TACHYON_CUDA
