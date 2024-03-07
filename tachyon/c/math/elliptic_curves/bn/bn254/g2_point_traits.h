#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G2_POINT_TRAITS_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G2_POINT_TRAITS_H_

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/cc/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon::cc::math {

template <>
struct PointTraits<tachyon::math::bn254::G2AffinePoint> {
  using CPoint = tachyon_bn254_g2_point2;
  using CCurvePoint = tachyon_bn254_g2_affine;
  using CScalarField = tachyon_bn254_fr;
};

template <>
struct PointTraits<tachyon::math::bn254::G2ProjectivePoint> {
  using CPoint = tachyon_bn254_g2_point3;
  using CCurvePoint = tachyon_bn254_g2_projective;
  using CScalarField = tachyon_bn254_fr;
};

template <>
struct PointTraits<tachyon::math::bn254::G2JacobianPoint> {
  using CPoint = tachyon_bn254_g2_point3;
  using CCurvePoint = tachyon_bn254_g2_jacobian;
  using CScalarField = tachyon_bn254_fr;
};

template <>
struct PointTraits<tachyon::math::bn254::G2PointXYZZ> {
  using CPoint = tachyon_bn254_g2_point4;
  using CCurvePoint = tachyon_bn254_g2_xyzz;
  using CScalarField = tachyon_bn254_fr;
};

template <>
struct PointTraits<tachyon_bn254_g2_affine> {
  using Point = tachyon::math::Point2<tachyon::math::bn254::Fq>;
  using CurvePoint = tachyon::math::bn254::G2AffinePoint;
};

template <>
struct PointTraits<tachyon_bn254_g2_projective> {
  using Point = tachyon::math::Point3<tachyon::math::bn254::Fq>;
  using CurvePoint = tachyon::math::bn254::G2ProjectivePoint;
};

template <>
struct PointTraits<tachyon_bn254_g2_jacobian> {
  using Point = tachyon::math::Point3<tachyon::math::bn254::Fq>;
  using CurvePoint = tachyon::math::bn254::G2JacobianPoint;
};

template <>
struct PointTraits<tachyon_bn254_g2_xyzz> {
  using Point = tachyon::math::Point4<tachyon::math::bn254::Fq>;
  using CurvePoint = tachyon::math::bn254::G2PointXYZZ;
};

template <>
struct PointTraits<tachyon_bn254_g2_point2> {
  using Point = tachyon::math::Point2<tachyon::math::bn254::Fq>;
};

template <>
struct PointTraits<tachyon_bn254_g2_point3> {
  using Point = tachyon::math::Point3<tachyon::math::bn254::Fq>;
};

template <>
struct PointTraits<tachyon_bn254_g2_point4> {
  using Point = tachyon::math::Point4<tachyon::math::bn254::Fq>;
};

}  // namespace tachyon::cc::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G1_POINT_TRAITS_H_
