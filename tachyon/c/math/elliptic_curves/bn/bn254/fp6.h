#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP6_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP6_H_

#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fp2.h"

struct tachyon_bn254_fp6 {
  tachyon_bn254_fp2 c0;
  tachyon_bn254_fp2 c1;
  tachyon_bn254_fp2 c2;
};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_bn254_fp6 tachyon_bn254_fp6_zero();

TACHYON_C_EXPORT tachyon_bn254_fp6 tachyon_bn254_fp6_one();

TACHYON_C_EXPORT tachyon_bn254_fp6 tachyon_bn254_fp6_random();

TACHYON_C_EXPORT tachyon_bn254_fp6
tachyon_bn254_fp6_dbl(const tachyon_bn254_fp6* a);

TACHYON_C_EXPORT tachyon_bn254_fp6
tachyon_bn254_fp6_neg(const tachyon_bn254_fp6* a);

TACHYON_C_EXPORT tachyon_bn254_fp6
tachyon_bn254_fp6_sqr(const tachyon_bn254_fp6* a);

TACHYON_C_EXPORT tachyon_bn254_fp6
tachyon_bn254_fp6_inv(const tachyon_bn254_fp6* a);

TACHYON_C_EXPORT tachyon_bn254_fp6
tachyon_bn254_fp6_add(const tachyon_bn254_fp6* a, const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT tachyon_bn254_fp6
tachyon_bn254_fp6_sub(const tachyon_bn254_fp6* a, const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT tachyon_bn254_fp6
tachyon_bn254_fp6_mul(const tachyon_bn254_fp6* a, const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT tachyon_bn254_fp6
tachyon_bn254_fp6_div(const tachyon_bn254_fp6* a, const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp6_eq(const tachyon_bn254_fp6* a,
                                           const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp6_ne(const tachyon_bn254_fp6* a,
                                           const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp6_gt(const tachyon_bn254_fp6* a,
                                           const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp6_ge(const tachyon_bn254_fp6* a,
                                           const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp6_lt(const tachyon_bn254_fp6* a,
                                           const tachyon_bn254_fp6* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp6_le(const tachyon_bn254_fp6* a,
                                           const tachyon_bn254_fp6* b);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP6_H_
