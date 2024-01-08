#ifndef TACHYON_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/point_conversions_forward.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"
#include "tachyon/math/elliptic_curves/projective_point.h"

namespace tachyon::math {

template <typename DstPoint, typename SrcPoint>
constexpr DstPoint ConvertPoint(const SrcPoint& src_point) {
  return PointConversions<SrcPoint, DstPoint>::Convert(src_point);
}

// TODO(insun35): Parallelize ConvertPoints using OpenMP.
template <typename DstContainer, typename SrcContainer>
[[nodiscard]] constexpr bool ConvertPoints(const SrcContainer& src_points,
                                           DstContainer* dst_points) {
  using DstPoint = typename DstContainer::value_type;

  if (std::size(src_points) != std::size(*dst_points)) return false;
  for (size_t i = 0; i < std::size(src_points); ++i) {
    (*dst_points)[i] = ConvertPoint<DstPoint>(src_points[i]);
  }
  return true;
}

template <typename Curve>
struct PointConversions<AffinePoint<Curve>, ProjectivePoint<Curve>> {
  constexpr static ProjectivePoint<Curve> Convert(
      const AffinePoint<Curve>& src_point) {
    return src_point.ToProjective();
  }
};

template <typename Curve>
struct PointConversions<AffinePoint<Curve>, JacobianPoint<Curve>> {
  constexpr static JacobianPoint<Curve> Convert(
      const AffinePoint<Curve>& src_point) {
    return src_point.ToJacobian();
  }
};

template <typename Curve>
struct PointConversions<AffinePoint<Curve>, PointXYZZ<Curve>> {
  constexpr static PointXYZZ<Curve> Convert(
      const AffinePoint<Curve>& src_point) {
    return src_point.ToXYZZ();
  }
};

template <typename Curve>
struct PointConversions<ProjectivePoint<Curve>, AffinePoint<Curve>> {
  constexpr static AffinePoint<Curve> Convert(
      const ProjectivePoint<Curve>& src_point) {
    return src_point.ToAffine();
  }
};

template <typename Curve>
struct PointConversions<ProjectivePoint<Curve>, JacobianPoint<Curve>> {
  constexpr static JacobianPoint<Curve> Convert(
      const ProjectivePoint<Curve>& src_point) {
    return src_point.ToJacobian();
  }
};

template <typename Curve>
struct PointConversions<ProjectivePoint<Curve>, PointXYZZ<Curve>> {
  constexpr static PointXYZZ<Curve> Convert(
      const ProjectivePoint<Curve>& src_point) {
    return src_point.ToXYZZ();
  }
};

template <typename Curve>
struct PointConversions<JacobianPoint<Curve>, AffinePoint<Curve>> {
  constexpr static AffinePoint<Curve> Convert(
      const JacobianPoint<Curve>& src_point) {
    return src_point.ToAffine();
  }
};

template <typename Curve>
struct PointConversions<JacobianPoint<Curve>, ProjectivePoint<Curve>> {
  constexpr static ProjectivePoint<Curve> Convert(
      const JacobianPoint<Curve>& src_point) {
    return src_point.ToProjective();
  }
};

template <typename Curve>
struct PointConversions<JacobianPoint<Curve>, PointXYZZ<Curve>> {
  constexpr static PointXYZZ<Curve> Convert(
      const JacobianPoint<Curve>& src_point) {
    return src_point.ToXYZZ();
  }
};

template <typename Curve>
struct PointConversions<PointXYZZ<Curve>, AffinePoint<Curve>> {
  constexpr static AffinePoint<Curve> Convert(
      const PointXYZZ<Curve>& src_point) {
    return src_point.ToAffine();
  }
};

template <typename Curve>
struct PointConversions<PointXYZZ<Curve>, ProjectivePoint<Curve>> {
  constexpr static ProjectivePoint<Curve> Convert(
      const PointXYZZ<Curve>& src_point) {
    return src_point.ToProjective();
  }
};

template <typename Curve>
struct PointConversions<PointXYZZ<Curve>, JacobianPoint<Curve>> {
  constexpr static JacobianPoint<Curve> Convert(
      const PointXYZZ<Curve>& src_point) {
    return src_point.ToJacobian();
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
