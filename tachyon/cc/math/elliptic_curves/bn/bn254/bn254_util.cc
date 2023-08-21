#include "tachyon/cc/math/elliptic_curves/bn/bn254/bn254_util.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

using namespace tachyon::math;

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_fq& fq) {
  BigInt<4> limbs(fq.limbs);
  return os << limbs;
}

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_fr& fr) {
  BigInt<4> limbs(fr.limbs);
  return os << limbs;
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_affine& point_in) {
  Point2<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  return os << bn254::G1AffinePoint(point,
                                    point.x.IsZero() && point.y.IsZero());
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_projective& point_in) {
  Point3<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  point.z = bn254::Fq::FromMontgomery(BigInt<4>(point_in.z.limbs));
  return os << bn254::G1ProjectivePoint(point);
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_jacobian& point_in) {
  Point3<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  point.z = bn254::Fq::FromMontgomery(BigInt<4>(point_in.z.limbs));
  return os << bn254::G1JacobianPoint(point);
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_xyzz& point_in) {
  Point4<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  point.z = bn254::Fq::FromMontgomery(BigInt<4>(point_in.zz.limbs));
  point.w = bn254::Fq::FromMontgomery(BigInt<4>(point_in.zzz.limbs));
  return os << bn254::G1PointXYZZ(point);
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_point2& point_in) {
  Point2<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  return os << point;
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_point3& point_in) {
  Point3<bn254::Fq> point;
  point.x = bn254::Fq::FromMontgomery(BigInt<4>(point_in.x.limbs));
  point.y = bn254::Fq::FromMontgomery(BigInt<4>(point_in.y.limbs));
  point.z = bn254::Fq::FromMontgomery(BigInt<4>(point_in.z.limbs));
  return os << point;
}
