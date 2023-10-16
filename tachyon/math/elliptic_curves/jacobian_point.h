#ifndef TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_

#include <utility>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/elliptic_curves/point_conversions_forward.h"
#include "tachyon/math/geometry/point3.h"

namespace tachyon {
namespace math {

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
    static_assert(SrcCurve::kIsSWCurve && DstCurve::kIsSWCurve);
    return JacobianPoint<DstCurve>::FromMontgomery(src_point.ToMontgomery());
  }
};

}  // namespace math

namespace base {

template <typename Curve>
class Copyable<math::JacobianPoint<Curve>> {
 public:
  static bool WriteTo(const math::JacobianPoint<Curve>& point, Buffer* buffer) {
    return buffer->WriteMany(point.x(), point.y(), point.z());
  }

  static bool ReadFrom(const Buffer& buffer,
                       math::JacobianPoint<Curve>* point) {
    using BaseField = typename math::JacobianPoint<Curve>::BaseField;
    BaseField x, y, z;
    if (!buffer.ReadMany(&x, &y, &z)) return false;

    *point =
        math::JacobianPoint<Curve>(std::move(x), std::move(y), std::move(z));
    return true;
  }

  static size_t EstimateSize(const math::JacobianPoint<Curve>& point) {
    return base::EstimateSize(point.x()) + base::EstimateSize(point.y()) +
           base::EstimateSize(point.z());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_JACOBIAN_POINT_H_
