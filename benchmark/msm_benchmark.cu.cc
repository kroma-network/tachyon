#if TACHYON_CUDA
#include <iostream>

#if defined(TACHYON_HAS_MATPLOTLIB)
#include "third_party/matplotlibcpp17/include/pyplot.h"

using namespace matplotlibcpp17;
#endif  // defined(TACHYON_HAS_MATPLOTLIB)

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/console/table_writer.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/base/time/time_interval.h"
#include "tachyon/device/gpu/cuda/scoped_memory.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/device/gpu/scoped_async_memory.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_cuda.cu.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_cuda.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/kernels/elliptic_curve_ops.cu.h"

namespace tachyon {

using namespace device;
using namespace math;

namespace {

std::vector<bn254::G1JacobianPoint> CreateRandomPoints(size_t nums) {
  std::vector<bn254::G1JacobianPoint> ret;
  ret.reserve(nums);
  bn254::G1JacobianPoint p = bn254::G1JacobianPoint::Curve::Generator();
  for (size_t i = 0; i < nums; ++i) {
    ret.push_back(p.DoubleInPlace());
  }
  return ret;
}

std::vector<bn254::Fr> CreateRandomScalars(size_t nums) {
  std::vector<bn254::Fr> ret;
  ret.reserve(nums);
  for (size_t i = 0; i < nums; ++i) {
    ret.push_back(bn254::Fr::Random());
  }
  return ret;
}

}  // namespace

int RealMain(int argc, char** argv) {
  std::vector<uint64_t> test_log_nums;
  base::FlagParser parser;
  parser.AddFlag<base::Flag<std::vector<uint64_t>>>(&test_log_nums)
      .set_short_name("-n")
      .set_required()
      .set_help("The log number of points to test");
  {
    std::string error;
    if (!parser.Parse(argc, argv, &error)) {
      tachyon_cerr << error << std::endl;
      return 1;
    }
  }

  bn254::G1AffinePointCuda::Curve::Init();
  bn254::G1AffinePoint::Curve::Init();
  VariableBaseMSMCuda<bn254::G1AffinePointCuda::Curve>::Setup();

  base::ranges::sort(test_log_nums);
  std::vector<std::string> names;
  std::vector<double> results;
  names.reserve(test_log_nums.size() * 2);
  results.reserve(test_log_nums.size() * 2);
  for (uint64_t test_log_num : test_log_nums) {
    names.push_back(absl::Substitute("CPU/$0", test_log_num));
  }
  for (uint64_t test_log_num : test_log_nums) {
    names.push_back(absl::Substitute("CUDA/$0", test_log_num));
  }

  std::vector<uint64_t> test_nums;
  for (uint64_t test_log_num : test_log_nums) {
    test_nums.push_back(1 << test_log_num);
  }

  std::cout << "Generating random points..." << std::endl;
  uint64_t max_nums = test_nums.back();
  std::vector<bn254::G1JacobianPoint> bases_cpu = CreateRandomPoints(max_nums);
  std::vector<bn254::Fr> scalars_cpu = CreateRandomScalars(max_nums);
  std::cout << "Generation completed" << std::endl;

  base::TimeInterval interval(base::TimeTicks::Now());
  std::vector<bn254::G1JacobianPoint> results_cpu;
  for (uint64_t test_num : test_nums) {
    auto bases_begin = bases_cpu.begin();
    auto scalars_begin = scalars_cpu.begin();
    results_cpu.push_back(VariableBaseMSM<bn254::G1JacobianPoint>::MSM(
        bases_begin, bases_begin + test_num, scalars_begin,
        scalars_begin + test_num));
    auto duration = interval.GetTimeDelta();
    std::cout << "calculate:" << duration.InSecondsF() << std::endl;
    results.push_back(duration.InSecondsF());
  }

  gpuError_t error = gpuDeviceReset();
  GPU_CHECK(error == gpuSuccess, error);
  gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                           gpuMemHandleTypeNone,
                           {gpuMemLocationTypeDevice, 0}};
  gpu::ScopedMemPool mem_pool = gpu::CreateMemPool(&props);
  uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
  error = gpuMemPoolSetAttribute(mem_pool.get(), gpuMemPoolAttrReleaseThreshold,
                                 &mem_pool_threshold);
  GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuMemPoolSetAttribute()";

  gpu::ScopedDeviceMemory<bn254::G1AffinePointCuda> d_bases =
      gpu::Malloc<bn254::G1AffinePointCuda>(test_nums.back());
  gpu::ScopedDeviceMemory<bn254::FrCuda> d_scalars =
      gpu::Malloc<bn254::FrCuda>(test_nums.back());
  gpu::ScopedDeviceMemory<bn254::G1JacobianPointCuda> d_results =
      gpu::Malloc<bn254::G1JacobianPointCuda>(256);
  std::unique_ptr<bn254::G1JacobianPoint[]> u_results(
      new bn254::G1JacobianPoint[256]);

  gpu::ScopedStream stream = gpu::CreateStream();
  kernels::msm::ExecutionConfig<bn254::G1AffinePointCuda::Curve> config;
  config.mem_pool = mem_pool.get();
  config.stream = stream.get();
  config.bases = d_bases.get();
  config.scalars = d_scalars.get();
  config.results = d_results.get();

  std::vector<math::bn254::G1JacobianPoint> results_gpu;
  std::vector<bn254::G1AffinePoint> bases_affine_cpu;
  bases_affine_cpu.reserve(bases_cpu.size());
  for (const bn254::G1JacobianPoint& p : bases_cpu) {
    bases_affine_cpu.push_back(p.ToAffine());
  }

  interval.Start();
  for (size_t i = 0; i < test_log_nums.size(); ++i) {
    auto now = base::TimeTicks::Now();
    gpuMemcpy(d_bases.get(), bases_affine_cpu.data(),
              sizeof(bn254::G1AffinePointCuda) * test_nums[i],
              gpuMemcpyHostToDevice);
    gpuMemcpy(d_scalars.get(), scalars_cpu.data(),
              sizeof(bn254::FrCuda) * test_nums[i], gpuMemcpyHostToDevice);
    std::cout << "host -> device:"
              << (base::TimeTicks::Now() - now).InSecondsF() << std::endl;
    config.log_scalars_count = test_log_nums[i];

    now = base::TimeTicks::Now();
    error = VariableBaseMSMCuda<bn254::G1AffinePointCuda::Curve>::ExecuteAsync(
        config);
    GPU_CHECK(error == gpuSuccess, error) << "Failed to ExecuteAsync()";
    error = gpuStreamSynchronize(stream.get());
    GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuStreamSynchronize()";
    std::cout << "calculate:" << (base::TimeTicks::Now() - now).InSecondsF()
              << std::endl;

    now = base::TimeTicks::Now();
    gpuMemcpy(u_results.get(), config.results,
              sizeof(bn254::G1JacobianPointCuda) * 256, gpuMemcpyDefault);
    std::cout << "device -> host:"
              << (base::TimeTicks::Now() - now).InSecondsF() << std::endl;
    bn254::G1JacobianPoint result = bn254::G1JacobianPoint::Zero();
    for (size_t i = 0; i < bn254::Fr::Config::kModulusBits; ++i) {
      size_t index = bn254::Fr::Config::kModulusBits - i - 1;
      bn254::G1JacobianPoint bucket = u_results[index];
      if (i == 0) {
        result = bucket;
      } else {
        result.DoubleInPlace();
        result += bucket;
      }
    }
    results_gpu.push_back(result);
    results.push_back(interval.GetTimeDelta().InSecondsF());
  }

  CHECK(results_cpu == results_gpu) << "Result not matched";

  base::TableWriterBuilder builder;
  base::TableWriter writer = builder.AlignHeaderLeft()
                                 .AddSpace(1)
                                 .FitToTerminalWidth()
                                 .StripTrailingAsciiWhitespace()
                                 .AddColumn("NAME")
                                 .AddColumn("TIME(sec)")
                                 .Build();
  for (size_t i = 0; i < results.size(); ++i) {
    writer.SetElement(i, 0, names[i]);
    writer.SetElement(i, 1, absl::StrCat(results[i]));
  }
  writer.Print(true);

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
