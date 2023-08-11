#ifndef TACHYON_MATH_ELLIPTIC_CURVES_POINT_TRAITS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_POINT_TRAITS_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"

namespace tachyon::math {

template <typename PointTy>
struct PointTraits {
  using AdditionResultTy = PointTy;
};

template <typename Config>
struct PointTraits<AffinePoint<Config>> {
  using AdditionResultTy = JacobianPoint<Config>;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_POINT_TRAITS_H_
