#ifndef TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_

#include <utility>

#include "tachyon/math/elliptic_curves/point_conversions_forward.h"
#include "tachyon/math/geometry/point3.h"

namespace tachyon::math {

template <typename Curve, typename SFINAE = void>
class JacobianPoint;

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
JacobianPoint<Curve> operator*(const ScalarField& v,
                               const JacobianPoint<Curve>& point) {
  return point * v;
}

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
    static_assert(SrcCurve::kType == DstCurve::kType);
    return JacobianPoint<DstCurve>::FromMontgomery(src_point.ToMontgomery());
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
