#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/c/export.h"

struct TACHYON_C_EXPORT tachyon_bn254_g1_affine {
  tachyon_bn254_fq x;
  tachyon_bn254_fq y;
  bool infinity = false;
};

struct TACHYON_C_EXPORT tachyon_bn254_g1_projective {
  tachyon_bn254_fq x;
  tachyon_bn254_fq y;
  tachyon_bn254_fq z;
};

struct TACHYON_C_EXPORT tachyon_bn254_g1_jacobian {
  tachyon_bn254_fq x;
  tachyon_bn254_fq y;
  tachyon_bn254_fq z;
};

struct TACHYON_C_EXPORT tachyon_bn254_g1_xyzz {
  tachyon_bn254_fq x;
  tachyon_bn254_fq y;
  tachyon_bn254_fq zz;
  tachyon_bn254_fq zzz;
};

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_
