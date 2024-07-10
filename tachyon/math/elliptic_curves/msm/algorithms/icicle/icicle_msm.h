#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_

#include <memory>

#include "third_party/icicle/include/fields/id.h"

#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bn254.h"
#include "tachyon/math/elliptic_curves/projective_point.h"

namespace tachyon::math {

template <typename Point>
class IcicleMSM {
 public:
  using Curve = typename Point::Curve;

  IcicleMSM(gpuMemPool_t mem_pool, gpuStream_t stream)
      : mem_pool_(mem_pool), stream_(stream) {
    device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
    config_.reset(new ::msm::MSMConfig{
        ctx,
        /*points_size=*/0,
        /*precompute_factor=*/1,
        /*c=*/0,
        /*bitsize=*/0,
        /*large_bucket_factor=*/10,
        /*batch_size=*/1,
        /*are_scalars_on_device=*/false,
        /*are_scalars_montgomery_form=*/true,
        // TODO(chokobole): Considering KZG commitment, bases can be loaded to
        // the device just once initially.
        /*are_points_on_device=*/false,
        /*are_points_montgomery_form=*/true,
        /*are_results_on_device=*/false,
        /*is_big_triangle=*/false,
        /*is_async=*/false});
    VLOG(1) << "IcicleMSM is created";
  }
  IcicleMSM(const IcicleMSM& other) = delete;
  IcicleMSM& operator=(const IcicleMSM& other) = delete;

  template <typename BaseContainer, typename ScalarContainer>
  [[nodiscard]] bool Run(const BaseContainer& cpu_bases,
                         const ScalarContainer& cpu_scalars,
                         ProjectivePoint<Curve>* cpu_result) {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif

    using BaseField = typename Point::BaseField;
    using BigInt = typename Point::BaseField::BigIntTy;

    size_t bases_size = std::size(cpu_bases);
    size_t scalars_size = std::size(cpu_scalars);

    if (bases_size != scalars_size) {
      LOG(ERROR) << "bases_size and scalars_size don't match";
      return false;
    }

    ::bn254::projective_t ret;
    gpuError_t error = tachyon_bn254_msm_cuda(
        reinterpret_cast<const ::bn254::scalar_t*>(std::data(cpu_scalars)),
        reinterpret_cast<const ::bn254::affine_t*>(std::data(cpu_bases)),
        bases_size, *config_, &ret);
    if (error != gpuSuccess) return false;
    *cpu_result = {BaseField(reinterpret_cast<const BigInt&>(ret.x)),
                   BaseField(reinterpret_cast<const BigInt&>(ret.y)),
                   BaseField(reinterpret_cast<const BigInt&>(ret.z))};
    return true;
  }

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  std::unique_ptr<::msm::MSMConfig> config_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
