// clang_format off
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fq_traits.h"
%{if IsG2}
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fq2_traits.h"
%{endif IsG2}
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fr.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fr_traits.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{g1_or_g2}.h"
#include "tachyon/c/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/%{g1_or_g2}.h"

namespace tachyon::c::math {

template <>
struct PointTraits<tachyon::math::%{type}::%{G1_or_G2}AffinePoint> {
  using CPoint = tachyon_%{type}_%{g1_or_g2}_point2;
  using CCurvePoint = tachyon_%{type}_%{g1_or_g2}_affine;
  using CScalarField = tachyon_%{type}_fr;
};

template <>
struct PointTraits<tachyon::math::%{type}::%{G1_or_G2}ProjectivePoint> {
  using CPoint = tachyon_%{type}_%{g1_or_g2}_point3;
  using CCurvePoint = tachyon_%{type}_%{g1_or_g2}_projective;
  using CScalarField = tachyon_%{type}_fr;
};

template <>
struct PointTraits<tachyon::math::%{type}::%{G1_or_G2}JacobianPoint> {
  using CPoint = tachyon_%{type}_%{g1_or_g2}_point3;
  using CCurvePoint = tachyon_%{type}_%{g1_or_g2}_jacobian;
  using CScalarField = tachyon_%{type}_fr;
};

template <>
struct PointTraits<tachyon::math::%{type}::%{G1_or_G2}PointXYZZ> {
  using CPoint = tachyon_%{type}_%{g1_or_g2}_point4;
  using CCurvePoint = tachyon_%{type}_%{g1_or_g2}_xyzz;
  using CScalarField = tachyon_%{type}_fr;
};

template <>
struct PointTraits<tachyon_%{type}_%{g1_or_g2}_affine> {
  using Point = tachyon::math::Point2<tachyon::math::%{type}::Fq>;
  using CurvePoint = tachyon::math::%{type}::%{G1_or_G2}AffinePoint;
};

template <>
struct PointTraits<tachyon_%{type}_%{g1_or_g2}_projective> {
  using Point = tachyon::math::Point3<tachyon::math::%{type}::Fq>;
  using CurvePoint = tachyon::math::%{type}::%{G1_or_G2}ProjectivePoint;
};

template <>
struct PointTraits<tachyon_%{type}_%{g1_or_g2}_jacobian> {
  using Point = tachyon::math::Point3<tachyon::math::%{type}::Fq>;
  using CurvePoint = tachyon::math::%{type}::%{G1_or_G2}JacobianPoint;
};

template <>
struct PointTraits<tachyon_%{type}_%{g1_or_g2}_xyzz> {
  using Point = tachyon::math::Point4<tachyon::math::%{type}::Fq>;
  using CurvePoint = tachyon::math::%{type}::%{G1_or_G2}PointXYZZ;
};

template <>
struct PointTraits<tachyon_%{type}_%{g1_or_g2}_point2> {
  using Point = tachyon::math::Point2<tachyon::math::%{type}::Fq>;
};

template <>
struct PointTraits<tachyon_%{type}_%{g1_or_g2}_point3> {
  using Point = tachyon::math::Point3<tachyon::math::%{type}::Fq>;
};

template <>
struct PointTraits<tachyon_%{type}_%{g1_or_g2}_point4> {
  using Point = tachyon::math::Point4<tachyon::math::%{type}::Fq>;
};

}  // namespace tachyon::cc::math
// clang_format off
