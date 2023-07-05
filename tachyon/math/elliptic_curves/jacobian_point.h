#ifndef TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_

#include <ostream>

namespace tachyon {
namespace math {

template <typename Config, typename SFINAE = void>
class JacobianPoint;

template <typename Config>
std::ostream& operator<<(std::ostream& os, const JacobianPoint<Config>& point) {
  return os << point.ToString();
}

template <typename Config, typename ScalarField = typename Config::ScalarField>
JacobianPoint<Config> operator*(const ScalarField& v,
                                const JacobianPoint<Config>& point) {
  return point * v;
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
