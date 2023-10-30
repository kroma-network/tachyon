#ifndef TACHYON_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_

#include <utility>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/elliptic_curves/point_conversions_forward.h"
#include "tachyon/math/geometry/point4.h"

namespace tachyon {
namespace math {

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
  static PointXYZZ<DstCurve> Convert(const PointXYZZ<SrcCurve>& src_point) {
    static_assert(SrcCurve::kType == DstCurve::kType);
    return PointXYZZ<DstCurve>::FromMontgomery(src_point.ToMontgomery());
  }
};

}  // namespace math

namespace base {

template <typename Curve>
class Copyable<math::PointXYZZ<Curve>> {
 public:
  static bool WriteTo(const math::PointXYZZ<Curve>& point, Buffer* buffer) {
    return buffer->WriteMany(point.x(), point.y(), point.zz(), point.zzz());
  }

  static bool ReadFrom(const Buffer& buffer, math::PointXYZZ<Curve>* point) {
    using BaseField = typename math::PointXYZZ<Curve>::BaseField;
    BaseField x, y, zz, zzz;
    if (!buffer.ReadMany(&x, &y, &zz, &zzz)) return false;

    *point = math::PointXYZZ<Curve>(std::move(x), std::move(y), std::move(zz),
                                    std::move(zzz));
    return true;
  }

  static size_t EstimateSize(const math::PointXYZZ<Curve>& point) {
    return base::EstimateSize(point.x()) + base::EstimateSize(point.y()) +
           base::EstimateSize(point.zz()) + base::EstimateSize(point.zzz());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_
