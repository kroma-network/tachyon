#ifndef TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_

#include <utility>

#include "tachyon/math/elliptic_curves/point_conversions_forward.h"

namespace tachyon::math {

template <typename Curve, typename SFINAE = void>
class AffinePoint;

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
auto operator*(const ScalarField& v, const AffinePoint<Curve>& point) {
  return point * v;
}

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
  static const AffinePoint<DstCurve>& Convert(
      const AffinePoint<SrcCurve>& src_point) {
    static_assert(SrcCurve::kType == DstCurve::kType);
    return reinterpret_cast<const AffinePoint<DstCurve>&>(src_point);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
