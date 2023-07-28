#ifndef TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_

#include <ostream>

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"

namespace tachyon::math {

template <typename Curve, typename SFINAE = void>
class JacobianPoint;

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const JacobianPoint<Curve>& point) {
  return os << point.ToString();
}

template <typename Curve, typename ScalarField = typename Curve::ScalarField>
JacobianPoint<Curve> operator*(const ScalarField& v,
                               const JacobianPoint<Curve>& point) {
  return point.ScalarMul(v.ToBigInt());
}

template <typename Curve>
class MultiplicativeIdentity<JacobianPoint<Curve>> {
 public:
  using P = JacobianPoint<Curve>;

  static const P& One() {
    static base::NoDestructor<P> one(P::One());
    return *one;
  }

  constexpr static bool IsOne(const P& value) { return value.IsOne(); }
};

template <typename Curve>
class AdditiveIdentity<JacobianPoint<Curve>> {
 public:
  using P = JacobianPoint<Curve>;

  static const P& Zero() {
    static base::NoDestructor<P> zero(P::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const P& value) { return value.IsZero(); }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
