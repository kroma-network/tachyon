#ifndef TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_

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

template <typename Config>
class MultiplicativeIdentity<AffinePoint<Config>> {
 public:
  using P = AffinePoint<Config>;

  static const P& One() {
    static base::NoDestructor<P> one(P::One());
    return *one;
  }

  constexpr static bool IsOne(const P& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<AffinePoint<Config>> {
 public:
  using P = AffinePoint<Config>;

  static const P& Zero() {
    static base::NoDestructor<P> zero(P::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const P& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
