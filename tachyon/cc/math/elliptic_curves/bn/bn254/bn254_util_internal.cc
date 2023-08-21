#include "tachyon/cc/math/elliptic_curves/bn/bn254/bn254_util_internal.h"

namespace tachyon::math {

BigInt<4> ToBigInt(const tachyon_bn254_fq& fq) { return BigInt<4>(fq.limbs); }

BigInt<4> ToBigInt(const tachyon_bn254_fr& fr) { return BigInt<4>(fr.limbs); }

bn254::G1AffinePoint ToAffinePoint(const tachyon_bn254_g1_affine& point_in) {
  Point2<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  return bn254::G1AffinePoint(point, point.x.IsZero() && point.y.IsZero());
}

bn254::G1ProjectivePoint ToProjectivePoint(
    const tachyon_bn254_g1_projective& point_in) {
  Point3<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  point.z = bn254::Fq::FromMontgomery(BigInt<4>(point_in.z.limbs));
  return bn254::G1ProjectivePoint(point);
}

bn254::G1JacobianPoint ToJacobianPoint(
    const tachyon_bn254_g1_jacobian& point_in) {
  Point3<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  point.z = bn254::Fq::FromMontgomery(BigInt<4>(point_in.z.limbs));
  return bn254::G1JacobianPoint(point);
}

bn254::G1PointXYZZ ToPointXYZZ(const tachyon_bn254_g1_xyzz& point_in) {
  Point4<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  point.z = bn254::Fq::FromMontgomery(BigInt<4>(point_in.zz.limbs));
  point.w = bn254::Fq::FromMontgomery(BigInt<4>(point_in.zzz.limbs));
  return bn254::G1PointXYZZ(point);
}

Point2<bn254::Fq> ToPoint2(const tachyon_bn254_point2& point_in) {
  Point2<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  return point;
}

Point3<bn254::Fq> ToPoint3(const tachyon_bn254_point3& point_in) {
  Point3<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  point.z = bn254::Fq::FromMontgomery(BigInt<4>(point_in.z.limbs));
  return point;
}

}  // namespace tachyon::math
