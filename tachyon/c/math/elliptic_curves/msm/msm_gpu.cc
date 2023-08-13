#include "tachyon/c/math/elliptic_curves/msm/msm_gpu.h"

#include "absl/types/span.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/device/gpu/cuda/scoped_memory.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1_cuda.cu.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_cuda.cu.h"

namespace tachyon {

using namespace device;
using namespace math;

namespace {

gpu::ScopedMemPool g_mem_pool;
gpu::ScopedStream g_stream;
gpu::ScopedDeviceMemory<bn254::G1AffinePointCuda> g_d_bases;
gpu::ScopedDeviceMemory<bn254::FrCuda> g_d_scalars;
gpu::ScopedDeviceMemory<bn254::G1JacobianPointCuda> g_d_results;
std::unique_ptr<bn254::G1JacobianPoint[]> g_u_results;

bn254::G1JacobianPoint DoMSMGpu(absl::Span<const bn254::G1AffinePoint> bases,
                                absl::Span<const bn254::Fr> scalars) {
  gpuMemcpy(g_d_bases.get(), bases.data(),
            sizeof(bn254::G1AffinePointCuda) * bases.size(),
            gpuMemcpyHostToDevice);
  gpuMemcpy(g_d_scalars.get(), scalars.data(),
            sizeof(bn254::FrCuda) * scalars.size(), gpuMemcpyHostToDevice);
  kernels::msm::ExecutionConfig<bn254::G1AffinePointCuda::Curve> config;
  config.mem_pool = g_mem_pool.get();
  config.stream = g_stream.get();
  config.bases = g_d_bases.get();
  config.scalars = g_d_scalars.get();
  config.results = g_d_results.get();
  config.log_scalars_count = base::bits::Log2Ceiling(scalars.size());

  bn254::G1JacobianPoint ret;
  GPU_MUST_SUCCESS(
      VariableBaseMSMCuda<bn254::G1AffinePointCuda::Curve>::Execute(
          config, g_u_results.get(), &ret),
      "Failed to Execute()");
  return ret;
}

void DoInitMSMGpu(uint8_t degree) {
  GPU_MUST_SUCCESS(gpuDeviceReset(), "Failed to gpuDeviceReset()");

  VariableBaseMSMCuda<bn254::G1AffinePointCuda::Curve>::Setup();

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
  g_d_bases = gpu::Malloc<bn254::G1AffinePointCuda>(size);
  g_d_scalars = gpu::Malloc<bn254::FrCuda>(size);
  g_d_results = gpu::Malloc<bn254::G1JacobianPointCuda>(256);
  g_u_results.reset(new bn254::G1JacobianPoint[256]);

  g_stream = gpu::CreateStream();
}

void DoReleaseMSMGpu() {
  g_d_bases.reset();
  g_d_scalars.reset();
  g_d_results.reset();
  g_u_results.reset();
  g_stream.reset();
  g_mem_pool.reset();
}

bn254::G1JacobianPoint DoMSMGpu(const tachyon_bn254_point2* bases_in,
                                size_t bases_len,
                                const tachyon_bn254_fr* scalars_in,
                                size_t scalars_len) {
  absl::Span<const Point2<bn254::Fq>> points(
      reinterpret_cast<const Point2<bn254::Fq>*>(bases_in), bases_len);
  std::vector<bn254::G1AffinePoint> bases =
      base::Map(points, [](const Point2<bn254::Fq>& point) {
        if (point.x.IsZero() && point.y.IsZero()) {
          return bn254::G1AffinePoint::Zero();
        }
        return bn254::G1AffinePoint(point);
      });
  absl::Span<const bn254::Fr> scalars(
      reinterpret_cast<const bn254::Fr*>(scalars_in), scalars_len);
  return DoMSMGpu(absl::MakeConstSpan(bases), scalars);
}

bn254::G1JacobianPoint DoMSMGpu(const tachyon_bn254_g1_affine* bases_in,
                                size_t bases_len,
                                const tachyon_bn254_fr* scalars_in,
                                size_t scalars_len) {
  absl::Span<const bn254::G1AffinePoint> bases(
      reinterpret_cast<const bn254::G1AffinePoint*>(bases_in), bases_len);
  absl::Span<const bn254::Fr> scalars(
      reinterpret_cast<const bn254::Fr*>(scalars_in), scalars_len);
  return DoMSMGpu(bases, scalars);
}

tachyon_bn254_g1_jacobian* ToCCPtr(const bn254::G1JacobianPoint& point) {
  tachyon_bn254_g1_jacobian* ret = new tachyon_bn254_g1_jacobian;
  memcpy(&ret->x, point.x().value().limbs, sizeof(uint64_t) * 4);
  memcpy(&ret->y, point.y().value().limbs, sizeof(uint64_t) * 4);
  memcpy(&ret->z, point.z().value().limbs, sizeof(uint64_t) * 4);
  return ret;
}

}  // namespace
}  // namespace tachyon

void tachyon_init_msm_gpu(uint8_t degree) { tachyon::DoInitMSMGpu(degree); }

void tachyon_release_msm_gpu() { tachyon::DoReleaseMSMGpu(); }

tachyon_bn254_g1_jacobian* tachyon_msm_g1_point2_gpu(
    const tachyon_bn254_point2* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::ToCCPtr(
      tachyon::DoMSMGpu(bases, bases_len, scalars, scalars_len));
}

tachyon_bn254_g1_jacobian* tachyon_msm_g1_affine_gpu(
    const tachyon_bn254_g1_affine* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::ToCCPtr(
      tachyon::DoMSMGpu(bases, bases_len, scalars, scalars_len));
}
