// clang-format off
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{g1_or_g2}.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/%{g1_or_g2}.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon_%{type}_%{g1_or_g2}_affine> {
  using NativeType = tachyon::math::%{type}::%{G1_or_G2}AffinePoint;
};

template <>
struct TypeTraits<tachyon::math::%{type}::%{G1_or_G2}AffinePoint> {
  using CType = tachyon_%{type}_%{g1_or_g2}_affine;
};

template <>
struct TypeTraits<tachyon_%{type}_%{g1_or_g2}_projective> {
  using NativeType = tachyon::math::%{type}::%{G1_or_G2}ProjectivePoint;
};

template <>
struct TypeTraits<tachyon::math::%{type}::%{G1_or_G2}ProjectivePoint> {
  using CType = tachyon_%{type}_%{g1_or_g2}_projective;
};

template <>
struct TypeTraits<tachyon_%{type}_%{g1_or_g2}_jacobian> {
  using NativeType = tachyon::math::%{type}::%{G1_or_G2}JacobianPoint;
};

template <>
struct TypeTraits<tachyon::math::%{type}::%{G1_or_G2}JacobianPoint> {
  using CType = tachyon_%{type}_%{g1_or_g2}_jacobian;
};

template <>
struct TypeTraits<tachyon_%{type}_%{g1_or_g2}_xyzz> {
  using NativeType = tachyon::math::%{type}::%{G1_or_G2}PointXYZZ;
};

template <>
struct TypeTraits<tachyon::math::%{type}::%{G1_or_G2}PointXYZZ> {
  using CType = tachyon_%{type}_%{g1_or_g2}_xyzz;
};

template <>
struct TypeTraits<tachyon_%{type}_%{g1_or_g2}_point2> {
  using NativeType = tachyon::math::Point2<tachyon::math::%{type}::%{Fq_or_Fq2}>;
};

template <>
struct TypeTraits<tachyon::math::Point2<tachyon::math::%{type}::%{Fq_or_Fq2}>> {
  using CType = tachyon_%{type}_%{g1_or_g2}_point2;
};

template <>
struct TypeTraits<tachyon_%{type}_%{g1_or_g2}_point3> {
  using NativeType = tachyon::math::Point3<tachyon::math::%{type}::%{Fq_or_Fq2}>;
};

template <>
struct TypeTraits<tachyon::math::Point3<tachyon::math::%{type}::%{Fq_or_Fq2}>> {
  using CType = tachyon_%{type}_%{g1_or_g2}_point3;
};

template <>
struct TypeTraits<tachyon_%{type}_%{g1_or_g2}_point4> {
  using NativeType = tachyon::math::Point4<tachyon::math::%{type}::%{Fq_or_Fq2}>;
};

template <>
struct TypeTraits<tachyon::math::Point4<tachyon::math::%{type}::%{Fq_or_Fq2}>> {
  using CType = tachyon_%{type}_%{g1_or_g2}_point4;
};

}  // namespace tachyon::c::base
// clang-format on
