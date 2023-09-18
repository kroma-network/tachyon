#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_

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
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_traits.h"

namespace tachyon::c::math {

template <typename GpuCurve>
struct MSMGpuApi {
  using GpuAffinePointTy = tachyon::math::AffinePoint<GpuCurve>;
  using GpuScalarField = typename GpuAffinePointTy::ScalarField;
  using CpuCurve = typename tachyon::math::SWCurveTraits<GpuCurve>::CpuCurve;
  using CpuAffinePointTy = tachyon::math::AffinePoint<CpuCurve>;

  tachyon::device::gpu::ScopedMemPool mem_pool;
  tachyon::device::gpu::ScopedStream stream;
  tachyon::device::gpu::GpuMemory<GpuAffinePointTy> d_bases;
  tachyon::device::gpu::GpuMemory<GpuScalarField> d_scalars;
  MSMInputProvider<CpuAffinePointTy> provider;
  std::unique_ptr<tachyon::math::VariableBaseMSMGpu<GpuCurve>> msm;

  std::string save_location;
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

    std::string_view save_location_str;
    if (base::Environment::Get("TACHYON_SAVE_LOCATION", &save_location_str)) {
      save_location = std::string(save_location_str);
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

    uint64_t size = static_cast<uint64_t>(1) << degree;
    d_bases = tachyon::device::gpu::GpuMemory<GpuAffinePointTy>::Malloc(size);
    d_scalars = tachyon::device::gpu::GpuMemory<GpuScalarField>::Malloc(size);

    stream = tachyon::device::gpu::CreateStream();
    provider.set_needs_align(true);
    msm.reset(new tachyon::math::VariableBaseMSMGpu<GpuCurve>(
        algorithm, mem_pool.get(), stream.get()));
  }
};

template <typename RetPointTy, typename GpuCurve, typename CPointTy,
          typename CScalarField,
          typename CRetPointTy =
              typename cc::math::PointTraits<RetPointTy>::CCurvePointTy,
          typename CpuCurve =
              typename tachyon::math::SWCurveTraits<GpuCurve>::CpuCurve>
CRetPointTy* DoMSMGpu(MSMGpuApi<GpuCurve>& msm_api, const CPointTy* bases,
                      const CScalarField* scalars, size_t size) {
  msm_api.provider.Inject(bases, scalars, size);

  size_t aligned_size = msm_api.provider.bases().size();
  CHECK(msm_api.d_bases.CopyFrom(msm_api.provider.bases().data(),
                                 tachyon::device::gpu::GpuMemoryType::kHost, 0,
                                 aligned_size));
  CHECK(msm_api.d_scalars.CopyFrom(msm_api.provider.scalars().data(),
                                   tachyon::device::gpu::GpuMemoryType::kHost,
                                   0, aligned_size));

  RetPointTy ret;
  CHECK(
      msm_api.msm->Run(msm_api.d_bases, msm_api.d_scalars, aligned_size, &ret));
  CRetPointTy* cret = new CRetPointTy();
  cc::math::ToCPoint3(ret, cret);

  if (msm_api.log_msm) {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Yellow();
    std::cout << "DoMSMGpu()" << msm_api.idx++ << std::endl;
    std::cout << ret.ToHexString() << std::endl;
  }

  if (!msm_api.save_location.empty()) {
    {
      std::vector<std::string> results;
      for (const auto& base : msm_api.provider.bases()) {
        results.push_back(base.ToMontgomery().ToString());
      }
      results.push_back("");
      base::WriteFile(
          base::FilePath(absl::Substitute(
              "$0/bases$1.txt", msm_api.save_location, msm_api.idx - 1)),
          absl::StrJoin(results, "\n"));
    }
    {
      std::vector<std::string> results;
      for (const auto& scalar : msm_api.provider.scalars()) {
        results.push_back(scalar.ToMontgomery().ToString());
      }
      results.push_back("");
      base::WriteFile(
          base::FilePath(absl::Substitute(
              "$0/scalars$1.txt", msm_api.save_location, msm_api.idx - 1)),
          absl::StrJoin(results, "\n"));
    }
  }

  return cret;
}

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_GPU_H_
