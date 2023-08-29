#include "tachyon/cc/math/elliptic_curves/bn/bn254/bn254_util.h"

#include "tachyon/cc/math/elliptic_curves/bn/bn254/bn254_util_internal.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

using namespace tachyon::math;

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_fq& fq) {
  return os << ToBigInt(fq);
}

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_fr& fr) {
  return os << ToBigInt(fr);
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_affine& point) {
  return os << ToAffinePoint(point);
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_projective& point) {
  return os << ToProjectivePoint(point);
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_jacobian& point) {
  return os << ToJacobianPoint(point);
}

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_g1_xyzz& point) {
  return os << ToPointXYZZ(point);
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_point2& point) {
  return os << ToPoint2(point);
}

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_point3& point) {
  return os << ToPoint3(point);
}

bool operator==(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b) {
  return ToBigInt(a) == ToBigInt(b);
}

bool operator!=(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b) {
  return ToBigInt(a) != ToBigInt(b);
}

bool operator<(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b) {
  return ToBigInt(a) < ToBigInt(b);
}

bool operator<=(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b) {
  return ToBigInt(a) <= ToBigInt(b);
}

bool operator>(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b) {
  return ToBigInt(a) > ToBigInt(b);
}

bool operator>=(const tachyon_bn254_fq& a, const tachyon_bn254_fq& b) {
  return ToBigInt(a) >= ToBigInt(b);
}

bool operator==(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b) {
  return ToBigInt(a) == ToBigInt(b);
}

bool operator!=(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b) {
  return ToBigInt(a) != ToBigInt(b);
}

bool operator<(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b) {
  return ToBigInt(a) < ToBigInt(b);
}

bool operator<=(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b) {
  return ToBigInt(a) <= ToBigInt(b);
}

bool operator>(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b) {
  return ToBigInt(a) > ToBigInt(b);
}

bool operator>=(const tachyon_bn254_fr& a, const tachyon_bn254_fr& b) {
  return ToBigInt(a) >= ToBigInt(b);
}

bool operator==(const tachyon_bn254_g1_affine& a,
                const tachyon_bn254_g1_affine& b) {
  return ToAffinePoint(a) == ToAffinePoint(b);
}

bool operator!=(const tachyon_bn254_g1_affine& a,
                const tachyon_bn254_g1_affine& b) {
  return ToAffinePoint(a) != ToAffinePoint(b);
}

bool operator==(const tachyon_bn254_g1_projective& a,
                const tachyon_bn254_g1_projective& b) {
  return ToProjectivePoint(a) == ToProjectivePoint(b);
}

bool operator!=(const tachyon_bn254_g1_projective& a,
                const tachyon_bn254_g1_projective& b) {
  return ToProjectivePoint(a) != ToProjectivePoint(b);
}

bool operator==(const tachyon_bn254_g1_jacobian& a,
                const tachyon_bn254_g1_jacobian& b) {
  return ToJacobianPoint(a) == ToJacobianPoint(b);
}

bool operator!=(const tachyon_bn254_g1_jacobian& a,
                const tachyon_bn254_g1_jacobian& b) {
  return ToJacobianPoint(a) != ToJacobianPoint(b);
}

bool operator==(const tachyon_bn254_g1_xyzz& a,
                const tachyon_bn254_g1_xyzz& b) {
  return ToPointXYZZ(a) == ToPointXYZZ(b);
}

bool operator!=(const tachyon_bn254_g1_xyzz& a,
                const tachyon_bn254_g1_xyzz& b) {
  return ToPointXYZZ(a) != ToPointXYZZ(b);
}

bool operator==(const tachyon_bn254_g1_point2& a,
                const tachyon_bn254_g1_point2& b) {
  return ToPoint2(a) == ToPoint2(b);
}

bool operator!=(const tachyon_bn254_g1_point2& a,
                const tachyon_bn254_g1_point2& b) {
  return ToPoint2(a) != ToPoint2(b);
}

bool operator==(const tachyon_bn254_g1_point3& a,
                const tachyon_bn254_g1_point3& b) {
  return ToPoint3(a) == ToPoint3(b);
}

bool operator!=(const tachyon_bn254_g1_point3& a,
                const tachyon_bn254_g1_point3& b) {
  return ToPoint3(a) != ToPoint3(b);
}
