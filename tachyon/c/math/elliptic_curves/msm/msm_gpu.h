#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_

#include <limits>
#include <memory>
#include <string>

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/base/console/console_stream.h"
#include "tachyon/base/environment.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/c/math/elliptic_curves/msm/algorithm.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_input_provider.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_gpu.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::c::math {

template <typename GpuCurve>
struct MSMGpuApi {
  using GpuAffinePoint = tachyon::math::AffinePoint<GpuCurve>;
  using GpuScalarField = typename GpuAffinePoint::ScalarField;
  using CpuCurve = typename GpuCurve::CpuCurve;
  using CpuAffinePoint = tachyon::math::AffinePoint<CpuCurve>;

  tachyon::device::gpu::ScopedMemPool mem_pool;
  tachyon::device::gpu::ScopedStream stream;
  tachyon::device::gpu::GpuMemory<GpuAffinePoint> d_bases;
  tachyon::device::gpu::GpuMemory<GpuScalarField> d_scalars;
  MSMInputProvider<CpuAffinePoint> provider;
  std::unique_ptr<tachyon::math::VariableBaseMSMGpu<GpuCurve>> msm;

  std::string msm_gpu_input_dir;
  bool log_msm = false;
  size_t idx = 0;

  MSMGpuApi(uint8_t degree, int algorithm_in) {
    tachyon::math::MSMAlgorithmKind algorithm;
    switch (algorithm_in) {
      case TACHYON_MSM_ALGO_BELLMAN_MSM:
        algorithm = tachyon::math::MSMAlgorithmKind::kBellmanMSM;
        break;
      case TACHYON_MSM_ALGO_CUZK:
        algorithm = tachyon::math::MSMAlgorithmKind::kCUZK;
        break;
      default:
        NOTREACHED() << "Not supported algorithm";
    }

    GPU_MUST_SUCCESS(gpuDeviceReset(), "Failed to gpuDeviceReset()");

    {
      // NOTE(chokobole): This should be replaced with VLOG().
      // Currently, there's no way to delegate VLOG flags from rust side.
      base::ConsoleStream cs;
      cs.Green();
      std::cout << "CreateMSMGpuApi()" << std::endl;
    }

    std::string_view msm_gpu_input_dir_str;
    if (base::Environment::Get("TACHYON_MSM_GPU_INPUT_DIR",
                               &msm_gpu_input_dir_str)) {
      msm_gpu_input_dir = std::string(msm_gpu_input_dir_str);
    }
    std::string_view log_msm_str;
    if (base::Environment::Get("TACHYON_LOG_MSM", &log_msm_str)) {
      if (log_msm_str == "1") log_msm = true;
    }

    gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                             gpuMemHandleTypeNone,
                             {gpuMemLocationTypeDevice, 0}};
    mem_pool = tachyon::device::gpu::CreateMemPool(&props);
    uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
    GPU_MUST_SUCCESS(
        gpuMemPoolSetAttribute(mem_pool.get(), gpuMemPoolAttrReleaseThreshold,
                               &mem_pool_threshold),
        "Failed to gpuMemPoolSetAttribute()");

    uint64_t size = uint64_t{1} << degree;
    d_bases = tachyon::device::gpu::GpuMemory<GpuAffinePoint>::Malloc(size);
    d_scalars = tachyon::device::gpu::GpuMemory<GpuScalarField>::Malloc(size);

    stream = tachyon::device::gpu::CreateStream();
    provider.set_needs_align(true);
    msm.reset(new tachyon::math::VariableBaseMSMGpu<GpuCurve>(
        algorithm, mem_pool.get(), stream.get()));
  }
};

template <
    typename RetPoint, typename GpuCurve, typename CPoint,
    typename CScalarField,
    typename CRetPoint = typename cc::math::PointTraits<RetPoint>::CCurvePoint,
    typename CpuCurve = typename GpuCurve::CpuCurve>
CRetPoint* DoMSMGpu(MSMGpuApi<GpuCurve>& msm_api, const CPoint* bases,
                    const CScalarField* scalars, size_t size) {
  msm_api.provider.Inject(bases, scalars, size);

  size_t aligned_size = msm_api.provider.bases().size();
  CHECK(msm_api.d_bases.CopyFrom(msm_api.provider.bases().data(),
                                 tachyon::device::gpu::GpuMemoryType::kHost, 0,
                                 aligned_size));
  CHECK(msm_api.d_scalars.CopyFrom(msm_api.provider.scalars().data(),
                                   tachyon::device::gpu::GpuMemoryType::kHost,
                                   0, aligned_size));

  RetPoint ret;
  CHECK(
      msm_api.msm->Run(msm_api.d_bases, msm_api.d_scalars, aligned_size, &ret));
  CRetPoint* cret = new CRetPoint();
  cc::math::ToCPoint3(ret, cret);

  if (msm_api.log_msm) {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Yellow();
    std::cout << "DoMSMGpu()" << msm_api.idx++ << std::endl;
    std::cout << ret.ToHexString() << std::endl;
  }

  if (!msm_api.msm_gpu_input_dir.empty()) {
    base::Uint8VectorBuffer buffer;
    {
      CHECK(buffer.Grow(base::EstimateSize(msm_api.provider.bases())));
      CHECK(buffer.Write(msm_api.provider.bases()));
      base::WriteFile(
          base::FilePath(absl::Substitute(
              "$0/bases$1.txt", msm_api.msm_gpu_input_dir, msm_api.idx - 1)),
          absl::MakeConstSpan(buffer.owned_buffer()));
    }
    {
      buffer.set_buffer_offset(0);
      CHECK(buffer.Grow(base::EstimateSize(msm_api.provider.scalars())));
      CHECK(buffer.Write(msm_api.provider.scalars()));
      base::WriteFile(
          base::FilePath(absl::Substitute(
              "$0/scalars$1.txt", msm_api.msm_gpu_input_dir, msm_api.idx - 1)),
          absl::MakeConstSpan(buffer.owned_buffer()));
    }
  }

  return cret;
}

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_
