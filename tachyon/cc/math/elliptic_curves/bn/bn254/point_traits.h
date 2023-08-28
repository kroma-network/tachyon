#ifndef TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_POINT_TRAITS_H_
#define TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_POINT_TRAITS_H_

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/cc/math/elliptic_curves/point_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::cc::math {

template <>
struct PointTraits<tachyon::math::bn254::G1AffinePoint> {
  using CPointTy = tachyon_bn254_g1_affine;
  using CScalarField = tachyon_bn254_fr;
};

template <>
struct PointTraits<tachyon::math::bn254::G1ProjectivePoint> {
  using CPointTy = tachyon_bn254_g1_projective;
  using CScalarField = tachyon_bn254_fr;
};

template <>
struct PointTraits<tachyon::math::bn254::G1JacobianPoint> {
  using CPointTy = tachyon_bn254_g1_jacobian;
  using CScalarField = tachyon_bn254_fr;
};

template <>
struct PointTraits<tachyon::math::bn254::G1PointXYZZ> {
  using CPointTy = tachyon_bn254_g1_xyzz;
  using CScalarField = tachyon_bn254_fr;
};

}  // namespace tachyon::cc::math

#endif  // TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_POINT_TRAITS_H_
