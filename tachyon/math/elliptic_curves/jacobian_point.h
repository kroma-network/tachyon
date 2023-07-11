#ifndef TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_

#include <ostream>

#include "third_party/gmp/include/gmpxx.h"

namespace tachyon {
namespace math {

template <typename Config, typename SFINAE = void>
class JacobianPoint;

template <typename Config>
std::ostream& operator<<(std::ostream& os, const JacobianPoint<Config>& point) {
  return os << point.ToString();
}

template <typename Config>
JacobianPoint<Config> operator*(const mpz_class& scalar,
                                const JacobianPoint<Config>& point) {
  return point.ScalarMul(scalar);
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
