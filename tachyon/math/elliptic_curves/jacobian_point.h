#ifndef TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_

#include <ostream>

#include "third_party/gmp/include/gmpxx.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"

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
  return point.ScalarMul(v.ToBigInt());
}

template <typename Config>
class MultiplicativeIdentity<JacobianPoint<Config>> {
 public:
  using P = JacobianPoint<Config>;

  static const P& One() {
    static base::NoDestructor<P> one(P::One());
    return *one;
  }

  constexpr static bool IsOne(const P& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<JacobianPoint<Config>> {
 public:
  using P = JacobianPoint<Config>;

  static const P& Zero() {
    static base::NoDestructor<P> zero(P::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const P& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
