#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_POINT2_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_POINT2_H_

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq.h"

struct TACHYON_C_EXPORT tachyon_bn254_point2 {
  tachyon_bn254_fq x;
  tachyon_bn254_fq y;
};

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_POINT2_H_
