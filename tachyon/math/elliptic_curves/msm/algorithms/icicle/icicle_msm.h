#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_

#include <memory>

#include "absl/types/span.h"
#include "third_party/icicle/include/msm/msm_config.h"

#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/g2.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/geometry/projective_point.h"

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
  using ScalarField = typename Point::ScalarField;

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

  [[nodiscard]] bool Run(absl::Span<const Point> bases,
                         absl::Span<const ScalarField> cpu_scalars,
                         ProjectivePoint<Curve>* cpu_result);

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  std::unique_ptr<::msm::MSMConfig> config_;
};

template <>
TACHYON_EXPORT bool IcicleMSM<bls12_381::G1AffinePoint>::Run(
    absl::Span<const bls12_381::G1AffinePoint> bases,
    absl::Span<const ScalarField> cpu_scalars,
    ProjectivePoint<Curve>* cpu_result);

template <>
TACHYON_EXPORT bool IcicleMSM<bls12_381::G2AffinePoint>::Run(
    absl::Span<const bls12_381::G2AffinePoint> bases,
    absl::Span<const ScalarField> cpu_scalars,
    ProjectivePoint<Curve>* cpu_result);

template <>
TACHYON_EXPORT bool IcicleMSM<bn254::G1AffinePoint>::Run(
    absl::Span<const bn254::G1AffinePoint> bases,
    absl::Span<const ScalarField> cpu_scalars,
    ProjectivePoint<Curve>* cpu_result);

template <>
TACHYON_EXPORT bool IcicleMSM<bn254::G2AffinePoint>::Run(
    absl::Span<const bn254::G2AffinePoint> bases,
    absl::Span<const ScalarField> cpu_scalars,
    ProjectivePoint<Curve>* cpu_result);

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
