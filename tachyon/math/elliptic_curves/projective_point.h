#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PROJECTIVE_POINT_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PROJECTIVE_POINT_H_

#include <utility>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/elliptic_curves/point_conversions_forward.h"
#include "tachyon/math/geometry/point3.h"

namespace tachyon {
namespace math {

template <typename Curve, typename SFINAE = void>
class ProjectivePoint;

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
ProjectivePoint<Curve> operator*(const ScalarField& v,
                                 const ProjectivePoint<Curve>& point) {
  return point * v;
}

template <typename Curve>
struct PointConversions<ProjectivePoint<Curve>, ProjectivePoint<Curve>> {
  constexpr static const ProjectivePoint<Curve>& Convert(
      const ProjectivePoint<Curve>& src_point) {
    return src_point;
  }
};

template <typename SrcCurve, typename DstCurve>
struct PointConversions<ProjectivePoint<SrcCurve>, ProjectivePoint<DstCurve>,
                        std::enable_if_t<!std::is_same_v<SrcCurve, DstCurve>>> {
  static ProjectivePoint<DstCurve> Convert(
      const ProjectivePoint<SrcCurve>& src_point) {
    static_assert(SrcCurve::kType == DstCurve::kType);
    return ProjectivePoint<DstCurve>::FromMontgomery(src_point.ToMontgomery());
  }
};

}  // namespace math

namespace base {

template <typename Curve>
class Copyable<math::ProjectivePoint<Curve>> {
 public:
  static bool WriteTo(const math::ProjectivePoint<Curve>& point,
                      Buffer* buffer) {
    return buffer->WriteMany(point.x(), point.y(), point.z());
  }

  static bool ReadFrom(const Buffer& buffer,
                       math::ProjectivePoint<Curve>* point) {
    using BaseField = typename math::ProjectivePoint<Curve>::BaseField;
    BaseField x, y, z;
    if (!buffer.ReadMany(&x, &y, &z)) return false;

    *point =
        math::ProjectivePoint<Curve>(std::move(x), std::move(y), std::move(z));
    return true;
  }

  static size_t EstimateSize(const math::ProjectivePoint<Curve>& point) {
    return base::EstimateSize(point.x()) + base::EstimateSize(point.y()) +
           base::EstimateSize(point.z());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PROJECTIVE_POINT_H_
