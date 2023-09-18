#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_

#include "absl/numeric/bits.h"
#include "absl/types/span.h"

#include "tachyon/cc/math/elliptic_curves/point_traits.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon::c::math {

template <typename AffinePointTy>
class MSMInputProvider {
 public:
  using BaseField = typename AffinePointTy::BaseField;
  using ScalarField = typename AffinePointTy::ScalarField;
  using CCurvePointTy =
      typename tachyon::cc::math::PointTraits<AffinePointTy>::CCurvePointTy;
  using CPointTy =
      typename tachyon::cc::math::PointTraits<AffinePointTy>::CPointTy;
  using CScalarField =
      typename tachyon::cc::math::PointTraits<AffinePointTy>::CScalarField;

  void set_needs_align(bool needs_align) { needs_align_ = needs_align; }

  absl::Span<const AffinePointTy> bases() const { return bases_; }
  absl::Span<const ScalarField> scalars() const { return scalars_; }

  void Clear() {
    bases_owned_.clear();
    scalars_owned_.clear();
  }

  void Inject(const CPointTy* bases_in, const CScalarField* scalars_in,
              size_t size) {
    absl::Span<const tachyon::math::Point2<BaseField>> points(
        reinterpret_cast<const tachyon::math::Point2<BaseField>*>(bases_in),
        size);
    size_t aligned_size = 0;
    if (needs_align_) {
      aligned_size = absl::bit_ceil(size);
      bases_owned_.resize(aligned_size);
      for (size_t i = size; i < aligned_size; ++i) {
        bases_owned_[i] = AffinePointTy::Zero();
      }
    } else {
      bases_owned_.resize(size);
    }
    for (size_t i = 0; i < size; ++i) {
      bases_owned_[i] = AffinePointTy(
          points[i], points[i].x.IsZero() && points[i].y.IsZero());
    }
    bases_ = absl::MakeConstSpan(bases_owned_);

    if (needs_align_) {
      scalars_owned_.resize(aligned_size);
      for (size_t i = 0; i < size; ++i) {
        scalars_owned_[i] = reinterpret_cast<const ScalarField*>(scalars_in)[i];
      }
      for (size_t i = size; i < aligned_size; ++i) {
        scalars_owned_[i] = ScalarField::Zero();
      }
      scalars_ = absl::MakeConstSpan(scalars_owned_);
    } else {
      scalars_ = absl::MakeConstSpan(
          reinterpret_cast<const ScalarField*>(scalars_in), size);
    }
  }

  void Inject(const CCurvePointTy* bases_in, const CScalarField* scalars_in,
              size_t size) {
    if (needs_align_) {
      size_t aligned_size = absl::bit_ceil(size);
      bases_owned_.resize(aligned_size);
      for (size_t i = 0; i < size; ++i) {
        bases_owned_[i] = reinterpret_cast<const AffinePointTy*>(bases_in)[i];
      }
      for (size_t i = size; i < aligned_size; ++i) {
        bases_owned_[i] = AffinePointTy::Zero();
      }
      scalars_owned_.resize(aligned_size);
      for (size_t i = 0; i < size; ++i) {
        scalars_owned_[i] = reinterpret_cast<const ScalarField*>(scalars_in)[i];
      }
      for (size_t i = size; i < aligned_size; ++i) {
        scalars_owned_[i] = ScalarField::Zero();
      }
      bases_ = absl::MakeConstSpan(bases_owned_);
      scalars_ = absl::MakeConstSpan(scalars_owned_);
    } else {
      bases_ = absl::MakeConstSpan(
          reinterpret_cast<const AffinePointTy*>(bases_in), size);
      scalars_ = absl::MakeConstSpan(
          reinterpret_cast<const ScalarField*>(scalars_in), size);
    }
  }

 private:
  bool needs_align_ = false;
  absl::Span<const AffinePointTy> bases_;
  absl::Span<const ScalarField> scalars_;
  std::vector<AffinePointTy> bases_owned_;
  std::vector<ScalarField> scalars_owned_;
};

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_
