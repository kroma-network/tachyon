// clang_format off
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fr.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1.h"
#include "tachyon/cc/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/g1.h"

namespace tachyon::cc::math {

template <>
struct PointTraits<tachyon::math::%{type}::G1AffinePoint> {
  using CPoint = tachyon_%{type}_g1_point2;
  using CCurvePoint = tachyon_%{type}_g1_affine;
  using CScalarField = tachyon_%{type}_fr;
};

template <>
struct PointTraits<tachyon::math::%{type}::G1ProjectivePoint> {
  using CPoint = tachyon_%{type}_g1_point3;
  using CCurvePoint = tachyon_%{type}_g1_projective;
  using CScalarField = tachyon_%{type}_fr;
};

template <>
struct PointTraits<tachyon::math::%{type}::G1JacobianPoint> {
  using CPoint = tachyon_%{type}_g1_point3;
  using CCurvePoint = tachyon_%{type}_g1_jacobian;
  using CScalarField = tachyon_%{type}_fr;
};

template <>
struct PointTraits<tachyon::math::%{type}::G1PointXYZZ> {
  using CPoint = tachyon_%{type}_g1_point4;
  using CCurvePoint = tachyon_%{type}_g1_xyzz;
  using CScalarField = tachyon_%{type}_fr;
};

template <>
struct PointTraits<tachyon_%{type}_g1_affine> {
  using Point = tachyon::math::Point2<tachyon::math::%{type}::Fq>;
  using CurvePoint = tachyon::math::%{type}::G1AffinePoint;
};

template <>
struct PointTraits<tachyon_%{type}_g1_projective> {
  using Point = tachyon::math::Point3<tachyon::math::%{type}::Fq>;
  using CurvePoint = tachyon::math::%{type}::G1ProjectivePoint;
};

template <>
struct PointTraits<tachyon_%{type}_g1_jacobian> {
  using Point = tachyon::math::Point3<tachyon::math::%{type}::Fq>;
  using CurvePoint = tachyon::math::%{type}::G1JacobianPoint;
};

template <>
struct PointTraits<tachyon_%{type}_g1_xyzz> {
  using Point = tachyon::math::Point4<tachyon::math::%{type}::Fq>;
  using CurvePoint = tachyon::math::%{type}::G1PointXYZZ;
};

template <>
struct PointTraits<tachyon_%{type}_g1_point2> {
  using Point = tachyon::math::Point2<tachyon::math::%{type}::Fq>;
};

template <>
struct PointTraits<tachyon_%{type}_g1_point3> {
  using Point = tachyon::math::Point3<tachyon::math::%{type}::Fq>;
};

template <>
struct PointTraits<tachyon_%{type}_g1_point4> {
  using Point = tachyon::math::Point4<tachyon::math::%{type}::Fq>;
};

}  // namespace tachyon::cc::math
// clang_format off
