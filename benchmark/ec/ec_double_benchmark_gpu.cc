#if TACHYON_CUDA || TACHYON_USE_ROCM
#include <iostream>

// clang-format off
#include "benchmark/ec/simple_ec_benchmark_reporter.h"
#include "benchmark/ec/ec_config.h"
// clang-format on
#include "tachyon/base/time/time_interval.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"
#include "tachyon/math/elliptic_curves/test/random.h"

namespace tachyon {

using namespace device;
using namespace math;

namespace {

// TODO(chokobole): Use openmp.
void TestDoubleOnCPU(const std::vector<math::bn254::G1AffinePoint>& bases,
                     std::vector<math::bn254::G1JacobianPoint>& results,
                     uint64_t nums) {
  for (uint64_t i = 0; i < nums; ++i) {
    results[i] = bases[i].Double();
  }
}

gpuError_t LaunchDouble(const math::bn254::G1AffinePointGpu* x,
                        math::bn254::G1JacobianPointGpu* y, uint64_t count) {
  math::kernels::Double<<<(count - 1) / 32 + 1, 32>>>(x, y, count);
  gpuError_t error = LOG_IF_GPU_LAST_ERROR("Failed to Double()");
  return error == gpuSuccess
             ? LOG_IF_GPU_ERROR(gpuDeviceSynchronize(),
                                "Failed to gpuDeviceSynchronize")
             : error;
}

void TestDoubleOnGPU(math::bn254::G1AffinePointGpu* bases_cuda,
                     math::bn254::G1JacobianPointGpu* results_cuda,
                     const std::vector<math::bn254::G1AffinePoint>& bases,
                     uint64_t nums) {
  for (uint64_t i = 0; i < nums; ++i) {
    bases_cuda[i] = ConvertPoint<math::bn254::G1AffinePointGpu>(bases[i]);
  }

  LaunchDouble(bases_cuda, results_cuda, nums);
}

}  // namespace

int RealMain(int argc, char** argv) {
  ECConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  math::bn254::G1AffinePointGpu::Curve::Init();
  math::bn254::G1AffinePoint::Curve::Init();

  SimpleECBenchmarkReporter reporter(config.point_nums());

  std::cout << "Generating random points..." << std::endl;
  uint64_t max_point_num = config.point_nums().back();
  std::vector<bn254::G1AffinePoint> bases =
      CreatePseudoRandomPoints<bn254::G1AffinePoint>(max_point_num);
  std::vector<bn254::Fr> scalars =
      base::CreateVector(max_point_num, []() { return bn254::Fr::Random(); });
  std::cout << "Generation completed" << std::endl;

  std::vector<math::bn254::G1JacobianPoint> results_cpu;
  results_cpu.resize(max_point_num);
  base::TimeInterval interval(base::TimeTicks::Now());
  for (uint64_t point_num : config.point_nums()) {
    TestDoubleOnCPU(bases, results_cpu, point_num);
    reporter.AddResult(interval.GetTimeDelta().InSecondsF());
  }

  GPU_MUST_SUCCESS(gpuDeviceReset(), "Failed to gpuDeviceReset()");
  auto bases_cuda =
      gpu::GpuMemory<math::bn254::G1AffinePointGpu>::MallocManaged(
          max_point_num);
  auto results_cuda =
      gpu::GpuMemory<math::bn254::G1JacobianPointGpu>::MallocManaged(
          max_point_num);

  interval.Reset();
  for (uint64_t point_num : config.point_nums()) {
    TestDoubleOnGPU(bases_cuda.get(), results_cuda.get(), bases, point_num);
    reporter.AddResult(interval.GetTimeDelta().InSecondsF());
  }

  reporter.Show();

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
#else
#include "tachyon/base/console/iostream.h"

int main(int argc, char **argv) {
  tachyon_cerr << "please build with --config cuda or --config rocm"
               << std::endl;
  return 1;
}
#endif  // TACHYON_CUDA
