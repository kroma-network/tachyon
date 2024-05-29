#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_

#include <limits>
#include <memory>
#include <string>

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/base/console/console_stream.h"
#include "tachyon/base/environment.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_input_provider.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm_gpu.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::c::math {

template <typename Point>
struct MSMGpuApi {
  using Curve = typename Point::Curve;

  tachyon::device::gpu::ScopedMemPool mem_pool;
  tachyon::device::gpu::ScopedStream stream;
  MSMInputProvider<Point> provider;
  std::unique_ptr<tachyon::math::VariableBaseMSMGpu<Point>> msm;

  std::string msm_gpu_input_dir;
  bool log_msm = false;
  size_t idx = 0;

  explicit MSMGpuApi(uint8_t degree) {
    GPU_MUST_SUCCESS(gpuDeviceReset(), "Failed to gpuDeviceReset()");

    {
      // NOTE(chokobole): This should be replaced with VLOG().
      // Currently, there's no way to delegate VLOG flags from rust side.
      tachyon::base::ConsoleStream cs;
      cs.Green();
      std::cout << "CreateMSMGpuApi()" << std::endl;
    }

    std::string_view msm_gpu_input_dir_str;
    if (tachyon::base::Environment::Get("TACHYON_MSM_GPU_INPUT_DIR",
                                        &msm_gpu_input_dir_str)) {
      msm_gpu_input_dir = std::string(msm_gpu_input_dir_str);
    }
    std::string_view log_msm_str;
    if (tachyon::base::Environment::Get("TACHYON_LOG_MSM", &log_msm_str)) {
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

    stream = tachyon::device::gpu::CreateStream();
    msm.reset(new tachyon::math::VariableBaseMSMGpu<Point>(mem_pool.get(),
                                                           stream.get()));
  }
};

template <typename RetPoint, typename Point, typename CPoint,
          typename CScalarField,
          typename CRetPoint = typename PointTraits<RetPoint>::CCurvePoint,
          typename Curve = typename Point::Curve>
CRetPoint* DoMSMGpu(MSMGpuApi<Point>& msm_api, const CPoint* bases,
                    const CScalarField* scalars, size_t size) {
  msm_api.provider.Inject(bases, scalars, size);

  tachyon::math::ProjectivePoint<Curve> ret;
  CHECK(msm_api.msm->Run(msm_api.provider.bases(), msm_api.provider.scalars(),
                         &ret));
  CRetPoint* cret = new CRetPoint();
  if constexpr (std::is_same_v<RetPoint,
                               tachyon::math::ProjectivePoint<Curve>>) {
    *cret = c::base::c_cast(ret);
  } else {
    RetPoint ret_tmp = tachyon::math::ConvertPoint<RetPoint>(ret);
    *cret = c::base::c_cast(ret_tmp);
  }

  if (msm_api.log_msm) {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    tachyon::base::ConsoleStream cs;
    cs.Yellow();
    std::cout << "DoMSMGpu()" << msm_api.idx++ << std::endl;
    std::cout << ret.ToHexString() << std::endl;
  }

  if (!msm_api.msm_gpu_input_dir.empty()) {
    tachyon::base::Uint8VectorBuffer buffer;
    {
      CHECK(buffer.Grow(tachyon::base::EstimateSize(msm_api.provider.bases())));
      CHECK(buffer.Write(msm_api.provider.bases()));
      tachyon::base::WriteFile(
          tachyon::base::FilePath(absl::Substitute(
              "$0/bases$1.txt", msm_api.msm_gpu_input_dir, msm_api.idx - 1)),
          buffer.owned_buffer());
    }
    {
      buffer.set_buffer_offset(0);
      CHECK(
          buffer.Grow(tachyon::base::EstimateSize(msm_api.provider.scalars())));
      CHECK(buffer.Write(msm_api.provider.scalars()));
      tachyon::base::WriteFile(
          tachyon::base::FilePath(absl::Substitute(
              "$0/scalars$1.txt", msm_api.msm_gpu_input_dir, msm_api.idx - 1)),
          buffer.owned_buffer());
    }
  }

  return cret;
}

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_
