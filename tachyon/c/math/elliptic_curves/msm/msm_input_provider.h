#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_

#include <vector>

#include "absl/numeric/bits.h"
#include "absl/types/span.h"

#include "tachyon/c/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon::c::math {

template <typename AffinePoint>
class MSMInputProvider {
 public:
  using ScalarField = typename AffinePoint::ScalarField;
  using CScalarField = typename PointTraits<AffinePoint>::CScalarField;

  absl::Span<const AffinePoint> bases() const { return bases_; }
  absl::Span<const ScalarField> scalars() const { return scalars_; }

  template <typename T>
  void Inject(const T* bases_in, const CScalarField* scalars_in, size_t size) {
    bases_ = absl::MakeConstSpan(reinterpret_cast<const AffinePoint*>(bases_in),
                                 size);
    scalars_ = absl::MakeConstSpan(
        reinterpret_cast<const ScalarField*>(scalars_in), size);
  }

 private:
  absl::Span<const AffinePoint> bases_;
  absl::Span<const ScalarField> scalars_;
};

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_
