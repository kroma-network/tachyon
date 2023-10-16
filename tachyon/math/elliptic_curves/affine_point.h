#ifndef TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_

#include <utility>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/point_conversions_forward.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon {
namespace math {

template <typename Curve, typename SFINAE = void>
class AffinePoint;

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
JacobianPoint<Curve> operator*(const ScalarField& v,
                               const AffinePoint<Curve>& point) {
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
  static AffinePoint<DstCurve> Convert(const AffinePoint<SrcCurve>& src_point) {
    static_assert(SrcCurve::kIsSWCurve && DstCurve::kIsSWCurve);
    return AffinePoint<DstCurve>::FromMontgomery(src_point.ToMontgomery());
  }
};

}  // namespace math

namespace base {

template <typename Curve>
class Copyable<math::AffinePoint<Curve>> {
 public:
  static bool WriteTo(const math::AffinePoint<Curve>& point, Buffer* buffer) {
    return buffer->WriteMany(point.x(), point.y(), point.infinity());
  }

  static bool ReadFrom(const Buffer& buffer, math::AffinePoint<Curve>* point) {
    using BaseField = typename math::AffinePoint<Curve>::BaseField;
    BaseField x, y;
    bool infinity;
    if (!buffer.ReadMany(&x, &y, &infinity)) return false;

    *point = math::AffinePoint<Curve>(std::move(x), std::move(y), infinity);
    return true;
  }

  static size_t EstimateSize(const math::AffinePoint<Curve>& point) {
    return base::EstimateSize(point.x()) + base::EstimateSize(point.y()) +
           base::EstimateSize(point.infinity());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
