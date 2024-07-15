#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_

#include <memory>

#include "third_party/icicle/include/fields/id.h"

#include "tachyon/base/bit_cast.h"
#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bn254_g1.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bn254_g2.h"
#include "tachyon/math/elliptic_curves/projective_point.h"

namespace tachyon::math {

struct TACHYON_EXPORT IcicleMSMOptions {
  int points_size = 0;
  int precompute_factor = 1;
  int c = 0;
  int bitsize = 0;
  int large_bucket_factor = 10;
  int batch_size = 1;
  bool are_scalars_on_device = false;
  bool are_scalars_montgomery_form = true;
  bool are_points_on_device = false;
  bool are_points_montgomery_form = true;
  bool are_results_on_device = false;
  bool is_big_triangle = false;
  bool is_async = false;
};

template <typename Point>
class IcicleMSM {
 public:
  using Curve = typename Point::Curve;

  IcicleMSM(gpuMemPool_t mem_pool, gpuStream_t stream,
            const IcicleMSMOptions& options = IcicleMSMOptions())
      : mem_pool_(mem_pool), stream_(stream) {
    ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
    config_.reset(new ::msm::MSMConfig{
        ctx,
        options.points_size,
        options.precompute_factor,
        options.c,
        options.bitsize,
        options.large_bucket_factor,
        options.batch_size,
        options.are_scalars_on_device,
        options.are_scalars_montgomery_form,
        // TODO(chokobole): Considering KZG commitment, bases can be loaded to
        // the device just once initially.
        options.are_points_on_device,
        options.are_points_montgomery_form,
        options.are_results_on_device,
        options.is_big_triangle,
        options.is_async,
    });
    VLOG(1) << "IcicleMSM is created";
  }
  IcicleMSM(const IcicleMSM& other) = delete;
  IcicleMSM& operator=(const IcicleMSM& other) = delete;

  template <typename BaseContainer, typename ScalarContainer>
  [[nodiscard]] bool Run(const BaseContainer& cpu_bases,
                         const ScalarContainer& cpu_scalars,
                         ProjectivePoint<Curve>* cpu_result);

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  std::unique_ptr<::msm::MSMConfig> config_;
};

template <>
template <typename BaseContainer, typename ScalarContainer>
bool IcicleMSM<bn254::G1AffinePoint>::Run(const BaseContainer& cpu_bases,
                                          const ScalarContainer& cpu_scalars,
                                          ProjectivePoint<Curve>* cpu_result) {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif

  size_t bases_size = std::size(cpu_bases);
  size_t scalars_size = std::size(cpu_scalars);

  if (bases_size != scalars_size) {
    LOG(ERROR) << "bases_size and scalars_size don't match";
    return false;
  }

  ::bn254::projective_t ret;
  gpuError_t error = tachyon_bn254_g1_msm_cuda(
      reinterpret_cast<const ::bn254::scalar_t*>(std::data(cpu_scalars)),
      reinterpret_cast<const ::bn254::affine_t*>(std::data(cpu_bases)),
      bases_size, *config_, &ret);
  if (error != gpuSuccess) return false;
  ret = ::bn254::projective_t::to_montgomery(ret);
  *cpu_result = base::bit_cast<ProjectivePoint<Curve>>(ret);
  return true;
}

template <>
template <typename BaseContainer, typename ScalarContainer>
bool IcicleMSM<bn254::G2AffinePoint>::Run(const BaseContainer& cpu_bases,
                                          const ScalarContainer& cpu_scalars,
                                          ProjectivePoint<Curve>* cpu_result) {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif

  size_t bases_size = std::size(cpu_bases);
  size_t scalars_size = std::size(cpu_scalars);

  if (bases_size != scalars_size) {
    LOG(ERROR) << "bases_size and scalars_size don't match";
    return false;
  }

  ::bn254::g2_projective_t ret;
  gpuError_t error = tachyon_bn254_g2_msm_cuda(
      reinterpret_cast<const ::bn254::scalar_t*>(std::data(cpu_scalars)),
      reinterpret_cast<const ::bn254::g2_affine_t*>(std::data(cpu_bases)),
      bases_size, *config_, &ret);
  if (error != gpuSuccess) return false;
  ret = ::bn254::g2_projective_t::to_montgomery(ret);
  *cpu_result = base::bit_cast<ProjectivePoint<Curve>>(ret);
  return true;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
