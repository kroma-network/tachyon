#ifndef TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_

#include "tachyon/math/elliptic_curves/jacobian_point.h"

namespace tachyon::math {

template <typename Curve, typename SFINAE = void>
class AffinePoint;

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const AffinePoint<Curve>& point) {
  return os << point.ToString();
}

template <typename Curve, typename ScalarField = typename Curve::ScalarField>
JacobianPoint<Curve> operator*(const ScalarField& v,
                               const AffinePoint<Curve>& point) {
  return point * v;
}

template <typename Curve>
class MultiplicativeIdentity<AffinePoint<Curve>> {
 public:
  using P = AffinePoint<Curve>;

  static const P& One() {
    static base::NoDestructor<P> one(P::One());
    return *one;
  }

  constexpr static bool IsOne(const P& value) { return value.IsOne(); }
};

template <typename Curve>
class AdditiveIdentity<AffinePoint<Curve>> {
 public:
  using P = AffinePoint<Curve>;

  static const P& Zero() {
    static base::NoDestructor<P> zero(P::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const P& value) { return value.IsZero(); }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
