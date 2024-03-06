#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP2_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP2_H_

#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq.h"

struct tachyon_bn254_fp2 {
  tachyon_bn254_fq c0;
  tachyon_bn254_fq c1;
};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_zero();

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_one();

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_random();

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_dbl(const tachyon_bn254_fp2* a);

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_neg(const tachyon_bn254_fp2* a);

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_sqr(const tachyon_bn254_fp2* a);

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_inv(const tachyon_bn254_fp2* a);

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_add(const tachyon_bn254_fp2* a,
                                            const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_sub(const tachyon_bn254_fp2* a,
                                            const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_mul(const tachyon_bn254_fp2* a,
                                            const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT tachyon_bn254_fp2 tachyon_bn254_fp2_div(const tachyon_bn254_fp2* a,
                                            const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp2_eq(const tachyon_bn254_fp2* a,
                                           const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp2_ne(const tachyon_bn254_fp2* a,
                                           const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp2_gt(const tachyon_bn254_fp2* a,
                                           const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp2_ge(const tachyon_bn254_fp2* a,
                                           const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp2_lt(const tachyon_bn254_fp2* a,
                                           const tachyon_bn254_fp2* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp2_le(const tachyon_bn254_fp2* a,
                                           const tachyon_bn254_fp2* b);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP2_H_
