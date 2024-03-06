#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP12_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP12_H_

#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fp6.h"

struct tachyon_bn254_fp12 {
  tachyon_bn254_fp6 c0;
  tachyon_bn254_fp6 c1;
};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_bn254_fp12 tachyon_bn254_fp12_zero();

TACHYON_C_EXPORT tachyon_bn254_fp12 tachyon_bn254_fp12_one();

TACHYON_C_EXPORT tachyon_bn254_fp12 tachyon_bn254_fp12_random();

TACHYON_C_EXPORT tachyon_bn254_fp12
tachyon_bn254_fp12_dbl(const tachyon_bn254_fp12* a);

TACHYON_C_EXPORT tachyon_bn254_fp12
tachyon_bn254_fp12_neg(const tachyon_bn254_fp12* a);

TACHYON_C_EXPORT tachyon_bn254_fp12
tachyon_bn254_fp12_sqr(const tachyon_bn254_fp12* a);

TACHYON_C_EXPORT tachyon_bn254_fp12
tachyon_bn254_fp12_inv(const tachyon_bn254_fp12* a);

TACHYON_C_EXPORT tachyon_bn254_fp12 tachyon_bn254_fp12_add(
    const tachyon_bn254_fp12* a, const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT tachyon_bn254_fp12 tachyon_bn254_fp12_sub(
    const tachyon_bn254_fp12* a, const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT tachyon_bn254_fp12 tachyon_bn254_fp12_mul(
    const tachyon_bn254_fp12* a, const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT tachyon_bn254_fp12 tachyon_bn254_fp12_div(
    const tachyon_bn254_fp12* a, const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp12_eq(const tachyon_bn254_fp12* a,
                                            const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp12_ne(const tachyon_bn254_fp12* a,
                                            const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp12_gt(const tachyon_bn254_fp12* a,
                                            const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp12_ge(const tachyon_bn254_fp12* a,
                                            const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp12_lt(const tachyon_bn254_fp12* a,
                                            const tachyon_bn254_fp12* b);

TACHYON_C_EXPORT bool tachyon_bn254_fp12_le(const tachyon_bn254_fp12* a,
                                            const tachyon_bn254_fp12* b);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FP12_H_
