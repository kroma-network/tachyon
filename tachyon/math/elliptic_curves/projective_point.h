#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PROJECTIVE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PROJECTIVE_POINT_H_

#include <ostream>

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"

namespace tachyon {
namespace math {

template <typename Curve, typename SFINAE = void>
class ProjectivePoint;

template <typename Curve>
std::ostream& operator<<(std::ostream& os,
                         const ProjectivePoint<Curve>& point) {
  return os << point.ToString();
}

template <typename Curve, typename ScalarField = typename Curve::ScalarField>
ProjectivePoint<Curve> operator*(const ScalarField& v,
                                 const ProjectivePoint<Curve>& point) {
  return point.ScalarMul(v.ToBigInt());
}

template <typename Curve>
class MultiplicativeIdentity<ProjectivePoint<Curve>> {
 public:
  using P = ProjectivePoint<Curve>;

  static const P& One() {
    static base::NoDestructor<P> one(P::One());
    return *one;
  }

  constexpr static bool IsOne(const P& value) { return value.IsOne(); }
};

template <typename Curve>
class AdditiveIdentity<ProjectivePoint<Curve>> {
 public:
  using P = ProjectivePoint<Curve>;

  static const P& Zero() {
    static base::NoDestructor<P> zero(P::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const P& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PROJECTIVE_POINT_H_
