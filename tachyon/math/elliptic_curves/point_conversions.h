#ifndef TACHYON_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_POINT_CONVERSIONS_H_

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"
#include "tachyon/math/elliptic_curves/projective_point.h"

namespace tachyon::math {

template <typename DstPointTy, typename SrcPointTy>
constexpr DstPointTy ConvertPoint(const SrcPointTy& src_point) {
  return PointConversions<SrcPointTy, DstPointTy>::Convert(src_point);
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
