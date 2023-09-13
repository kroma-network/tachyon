#ifndef TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_

#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/point_conversions_forward.h"

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

template <typename Curve>
struct PointConversions<AffinePoint<Curve>, AffinePoint<Curve>> {
  constexpr static const AffinePoint<Curve>& Convert(
      const AffinePoint<Curve>& src_point) {
    return src_point;
  }
};

template <typename SrcCurve, typename DstCurve>
struct PointConversions<AffinePoint<SrcCurve>, AffinePoint<DstCurve>,
                        std::enable_if_t<!std::is_same_v<SrcCurve, DstCurve>>> {
  static AffinePoint<DstCurve> Convert(const AffinePoint<SrcCurve>& src_point) {
    static_assert(SrcCurve::kIsSWCurve && DstCurve::kIsSWCurve);
    return AffinePoint<DstCurve>::FromMontgomery(src_point.ToMontgomery());
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
