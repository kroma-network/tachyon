#ifndef TACHYON_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_

#include <utility>

#include "tachyon/math/elliptic_curves/point_conversions_forward.h"
#include "tachyon/math/geometry/point4.h"

namespace tachyon::math {

template <typename Curve, typename SFINAE = void>
class PointXYZZ;

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
PointXYZZ<Curve> operator*(const ScalarField& v,
                           const PointXYZZ<Curve>& point) {
  return point * v;
}

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
  static const PointXYZZ<DstCurve>& Convert(
      const PointXYZZ<SrcCurve>& src_point) {
    static_assert(SrcCurve::kType == DstCurve::kType);
    return reinterpret_cast<const PointXYZZ<DstCurve>&>(src_point);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_
