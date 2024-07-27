#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SEMIGROUPS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SEMIGROUPS_H_

#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/geometry/affine_point.h"
#include "tachyon/math/geometry/jacobian_point.h"

namespace tachyon::math::internal {

template <typename Curve>
struct AdditiveSemigroupTraits<AffinePoint<Curve>> {
  using ReturnTy = JacobianPoint<Curve>;
};

}  // namespace tachyon::math::internal

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SEMIGROUPS_H_
