#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_

#include <vector>

#include "absl/numeric/bits.h"
#include "absl/types/span.h"

#include "tachyon/base/openmp_util.h"
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon::c::math {

template <typename AffinePoint>
class MSMInputProvider {
 public:
  using ScalarField = typename AffinePoint::ScalarField;
  using CScalarField = typename PointTraits<AffinePoint>::CScalarField;

  void set_needs_align(bool needs_align) { needs_align_ = needs_align; }

  absl::Span<const AffinePoint> bases() const { return bases_; }
  absl::Span<const ScalarField> scalars() const { return scalars_; }

  void Clear() {
    bases_owned_.clear();
    scalars_owned_.clear();
  }

  template <typename Base>
  void Inject(const Base* bases_in, const CScalarField* scalars_in,
              size_t size) {
    if (needs_align_) {
      size_t aligned_size = absl::bit_ceil(size);
      bases_owned_.resize(aligned_size);
      scalars_owned_.resize(aligned_size);
      OPENMP_PARALLEL_FOR(size_t i = 0; i < aligned_size; ++i) {
        if (i < size) {
          bases_owned_[i] = reinterpret_cast<const AffinePoint*>(bases_in)[i];
          scalars_owned_[i] = base::native_cast(scalars_in)[i];
        } else {
          bases_owned_[i] = AffinePoint::Zero();
          scalars_owned_[i] = ScalarField::Zero();
        }
      }
      bases_ = bases_owned_;
      scalars_ = scalars_owned_;
    } else {
      bases_ = absl::MakeConstSpan(
          reinterpret_cast<const AffinePoint*>(bases_in), size);
      scalars_ = absl::MakeConstSpan(base::native_cast(scalars_in), size);
    }
  }

 private:
  bool needs_align_ = false;
  absl::Span<const AffinePoint> bases_;
  absl::Span<const ScalarField> scalars_;
  std::vector<AffinePoint> bases_owned_;
  std::vector<ScalarField> scalars_owned_;
};

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_
