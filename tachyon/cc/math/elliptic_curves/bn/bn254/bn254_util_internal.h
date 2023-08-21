#ifndef TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_INTERNAL_H_
#define TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_INTERNAL_H_

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/point2.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/point3.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::math {

BigInt<4> ToBigInt(const tachyon_bn254_fq& fq);

BigInt<4> ToBigInt(const tachyon_bn254_fr& fr);

bn254::G1AffinePoint ToAffinePoint(const tachyon_bn254_g1_affine& point_in);

bn254::G1ProjectivePoint ToProjectivePoint(
    const tachyon_bn254_g1_projective& point_in);

bn254::G1JacobianPoint ToJacobianPoint(
    const tachyon_bn254_g1_jacobian& point_in);

bn254::G1PointXYZZ ToPointXYZZ(const tachyon_bn254_g1_xyzz& point_in);

Point2<bn254::Fq> ToPoint2(const tachyon_bn254_point2& point_in);

Point3<bn254::Fq> ToPoint3(const tachyon_bn254_point3& point_in);

}  // namespace tachyon::math

#endif  // TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_INTERNAL_H_
