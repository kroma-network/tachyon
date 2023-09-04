#include "tachyon/c/math/elliptic_curves/msm/msm_gpu.h"

#include "absl/types/span.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/console/console_stream.h"
#include "tachyon/base/environment.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_input_provider.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_cuda.cu.h"

namespace tachyon {

using namespace math;
using namespace device;

namespace c::math {

namespace {

gpu::ScopedMemPool g_mem_pool;
gpu::ScopedStream g_stream;
gpu::GpuMemory<bn254::G1AffinePointGpu> g_d_bases;
gpu::GpuMemory<bn254::FrCuda> g_d_scalars;
gpu::GpuMemory<bn254::G1JacobianPointGpu> g_d_results;
std::unique_ptr<bn254::G1JacobianPoint[]> g_u_results;
std::unique_ptr<MSMInputProvider> g_provider;

// TODO(chokobole): Remove this when MSM gpu is stabilized.
std::string g_save_location;
bool g_log_msm = false;

size_t g_idx = 0;

void DoInitMSMGpu(uint8_t degree) {
  {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Green();
    std::cout << "DoInitMSMGpu()" << std::endl;
  }
  GPU_MUST_SUCCESS(gpuDeviceReset(), "Failed to gpuDeviceReset()");

  std::string_view save_location_str;
  if (base::Environment::Get("TACHYON_SAVE_LOCATION", &save_location_str)) {
    g_save_location = std::string(save_location_str);
  }
  std::string_view log_msm_str;
  if (base::Environment::Get("TACHYON_LOG_MSM", &log_msm_str)) {
    if (log_msm_str == "1") g_log_msm = true;
  }

  bn254::G1AffinePointGpu::Curve::Init();
  VariableBaseMSMCuda<bn254::G1AffinePointGpu::Curve>::Setup();

  gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                           gpuMemHandleTypeNone,
                           {gpuMemLocationTypeDevice, 0}};
  g_mem_pool = gpu::CreateMemPool(&props);
  uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
  GPU_MUST_SUCCESS(
      gpuMemPoolSetAttribute(g_mem_pool.get(), gpuMemPoolAttrReleaseThreshold,
                             &mem_pool_threshold),
      "Failed to gpuMemPoolSetAttribute()");

  uint64_t size = static_cast<uint64_t>(1) << degree;
  g_d_bases = gpu::GpuMemory<bn254::G1AffinePointGpu>::Malloc(size);
  g_d_scalars = gpu::GpuMemory<bn254::FrCuda>::Malloc(size);
  size_t bit_size = bn254::FrCuda::kModulusBits;
  g_d_results = gpu::GpuMemory<bn254::G1JacobianPointGpu>::Malloc(bit_size);
  g_u_results.reset(new bn254::G1JacobianPoint[bit_size]);

  g_stream = gpu::CreateStream();
  g_provider.reset(new MSMInputProvider());
  g_provider->set_needs_align(true);
}

void DoReleaseMSMGpu() {
  {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Green();
    std::cout << "DoReleaseMSMGpu()" << std::endl;
  }
  g_d_bases.reset();
  g_d_scalars.reset();
  g_d_results.reset();
  g_u_results.reset();
  g_stream.reset();
  g_mem_pool.reset();
  g_provider.reset();
}

bn254::G1JacobianPoint DoMSMGpuInternal(
    absl::Span<const bn254::G1AffinePoint> bases,
    absl::Span<const bn254::Fr> scalars) {
  CHECK(g_d_bases.CopyFrom(bases.data(), gpu::GpuMemoryType::kHost));
  CHECK(g_d_scalars.CopyFrom(scalars.data(), gpu::GpuMemoryType::kHost));

  msm::ExecutionConfig<bn254::G1AffinePointGpu::Curve> config;
  config.mem_pool = g_mem_pool.get();
  config.stream = g_stream.get();
  config.bases = g_d_bases.get();
  config.scalars = g_d_scalars.get();
  config.results = g_d_results.get();
  config.log_scalars_count = base::bits::Log2Ceiling(scalars.size());

  bn254::G1JacobianPoint ret;
  GPU_MUST_SUCCESS(VariableBaseMSMCuda<bn254::G1AffinePointGpu::Curve>::Execute(
                       config, g_u_results.get(), &ret),
                   "Failed to Execute()");
  if (g_log_msm) {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Yellow();
    std::cout << "DoMSMGpuInternal()" << g_idx++ << std::endl;
    std::cout << ret.ToHexString() << std::endl;
  }
  if (!g_save_location.empty()) {
    {
      std::vector<std::string> results;
      for (const bn254::G1AffinePoint& base : bases) {
        results.push_back(base.ToMontgomery().ToString());
      }
      results.push_back("");
      base::WriteFile(base::FilePath(absl::Substitute(
                          "$0/bases$1.txt", g_save_location, g_idx - 1)),
                      absl::StrJoin(results, "\n"));
    }
    {
      std::vector<std::string> results;
      for (const bn254::Fr& scalar : scalars) {
        results.push_back(scalar.ToMontgomery().ToString());
      }
      results.push_back("");
      base::WriteFile(base::FilePath(absl::Substitute(
                          "$0/scalars$1.txt", g_save_location, g_idx - 1)),
                      absl::StrJoin(results, "\n"));
    }
  }
  return ret;
}

template <typename T>
tachyon_bn254_g1_jacobian* DoMSMGpu(const T* bases, size_t bases_len,
                                    const tachyon_bn254_fr* scalars,
                                    size_t scalars_len) {
  g_provider->Inject(bases, bases_len, scalars, scalars_len);
  return CreateCPoint3Ptr<tachyon_bn254_g1_jacobian>(
      DoMSMGpuInternal(g_provider->bases(), g_provider->scalars()));
}

}  // namespace

}  // namespace c::math
}  // namespace tachyon

void tachyon_init_msm_gpu(uint8_t degree) {
  tachyon::c::math::DoInitMSMGpu(degree);
}

void tachyon_release_msm_gpu() { tachyon::c::math::DoReleaseMSMGpu(); }

tachyon_bn254_g1_jacobian* tachyon_bn254_g1_point2_msm_gpu(
    const tachyon_bn254_g1_point2* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::c::math::DoMSMGpu(bases, bases_len, scalars, scalars_len);
}

tachyon_bn254_g1_jacobian* tachyon_msm_g1_affine_gpu(
    const tachyon_bn254_g1_affine* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::c::math::DoMSMGpu(bases, bases_len, scalars, scalars_len);
}
