#ifndef TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_

#include <ostream>

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/elliptic_curves/point_conversions_forward.h"

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
  return point * v;
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

template <typename Curve>
struct PointConversions<JacobianPoint<Curve>, JacobianPoint<Curve>> {
  constexpr static const JacobianPoint<Curve>& Convert(
      const JacobianPoint<Curve>& src_point) {
    return src_point;
  }
};

template <typename SrcCurve, typename DstCurve>
struct PointConversions<JacobianPoint<SrcCurve>, JacobianPoint<DstCurve>,
                        std::enable_if_t<!std::is_same_v<SrcCurve, DstCurve>>> {
  static JacobianPoint<DstCurve> Convert(
      const JacobianPoint<SrcCurve>& src_point) {
    static_assert(SrcCurve::kIsSWCurve && DstCurve::kIsSWCurve);
    return JacobianPoint<DstCurve>::FromMontgomery(src_point.ToMontgomery());
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
