#ifndef TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_

#include <ostream>

#include "tachyon/math/elliptic_curves/jacobian_point.h"

namespace tachyon {
namespace math {

template <typename Config, typename SFINAE = void>
class AffinePoint;

template <typename Config>
std::ostream& operator<<(std::ostream& os, const AffinePoint<Config>& point) {
  return os << point.ToString();
}

template <typename Config, typename ScalarField = typename Config::ScalarField>
JacobianPoint<Config> operator*(const ScalarField& v,
                                const AffinePoint<Config>& point) {
  return point * v;
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
