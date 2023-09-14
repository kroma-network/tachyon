#ifndef TACHYON_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_

#include <ostream>

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/elliptic_curves/point_conversions_forward.h"

namespace tachyon::math {

template <typename Curve, typename SFINAE = void>
class PointXYZZ;

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const PointXYZZ<Curve>& point) {
  return os << point.ToString();
}

template <typename Curve, typename ScalarField = typename Curve::ScalarField>
PointXYZZ<Curve> operator*(const ScalarField& v,
                           const PointXYZZ<Curve>& point) {
  return point * v;
}

template <typename Curve>
class MultiplicativeIdentity<PointXYZZ<Curve>> {
 public:
  using P = PointXYZZ<Curve>;

  static const P& One() {
    static base::NoDestructor<P> one(P::One());
    return *one;
  }

  constexpr static bool IsOne(const P& value) { return value.IsOne(); }
};

template <typename Curve>
class AdditiveIdentity<PointXYZZ<Curve>> {
 public:
  using P = PointXYZZ<Curve>;

  static const P& Zero() {
    static base::NoDestructor<P> zero(P::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const P& value) { return value.IsZero(); }
};

template <typename Curve>
struct PointConversions<PointXYZZ<Curve>, PointXYZZ<Curve>> {
  constexpr static const PointXYZZ<Curve>& Convert(
      const PointXYZZ<Curve>& src_point) {
    return src_point;
  }
};

template <typename SrcCurve, typename DstCurve>
struct PointConversions<PointXYZZ<SrcCurve>, PointXYZZ<DstCurve>,
                        std::enable_if_t<!std::is_same_v<SrcCurve, DstCurve>>> {
  static PointXYZZ<DstCurve> Convert(const PointXYZZ<SrcCurve>& src_point) {
    static_assert(SrcCurve::kIsSWCurve && DstCurve::kIsSWCurve);
    return PointXYZZ<DstCurve>::FromMontgomery(src_point.ToMontgomery());
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
