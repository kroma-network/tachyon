#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_

#include <memory>

#include "third_party/icicle/include/fields/id.h"

#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bn254.h"
#include "tachyon/math/elliptic_curves/projective_point.h"

namespace tachyon::math {

template <typename GpuCurve>
class IcicleMSM {
 public:
  using ScalarField = typename AffinePoint<GpuCurve>::ScalarField;
  using CpuCurve = typename GpuCurve::CpuCurve;

  IcicleMSM(gpuMemPool_t mem_pool, gpuStream_t stream)
      : mem_pool_(mem_pool), stream_(stream) {
    device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
    config_.reset(new ::msm::MSMConfig{ctx,
                                       /*points_size=*/0,
                                       /*precompute_factor=*/1,
                                       /*c=*/0,
                                       /*bitsize=*/0,
                                       /*large_bucket_factor=*/10,
                                       /*batch_size=*/1,
                                       /*are_scalars_on_device=*/true,
                                       /*are_scalars_montgomery_form=*/true,
                                       /*are_points_on_device=*/true,
                                       /*are_points_montgomery_form=*/true,
                                       /*are_results_on_device=*/false,
                                       /*is_big_triangle=*/false,
                                       /*is_async=*/false});
  }
  IcicleMSM(const IcicleMSM& other) = delete;
  IcicleMSM& operator=(const IcicleMSM& other) = delete;

  [[nodiscard]] bool Run(
      const device::gpu::GpuMemory<AffinePoint<GpuCurve>>& bases,
      const device::gpu::GpuMemory<ScalarField>& scalars, size_t size,
      ProjectivePoint<CpuCurve>* cpu_result) {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif

    using CpuBaseField = typename ProjectivePoint<CpuCurve>::BaseField;
    using BigInt = typename ProjectivePoint<CpuCurve>::BaseField::BigIntTy;

    ::bn254::projective_t ret;
    gpuError_t error = tachyon_bn254_msm_cuda(
        reinterpret_cast<const ::bn254::scalar_t*>(scalars.get()),
        reinterpret_cast<const ::bn254::affine_t*>(bases.get()), size, *config_,
        &ret);
    if (error != gpuSuccess) return false;
    *cpu_result = {CpuBaseField(reinterpret_cast<const BigInt&>(ret.x)),
                   CpuBaseField(reinterpret_cast<const BigInt&>(ret.y)),
                   CpuBaseField(reinterpret_cast<const BigInt&>(ret.z))};
    return true;
  }

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  std::unique_ptr<::msm::MSMConfig> config_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
