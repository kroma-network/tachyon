#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_DENSE_POLYNOMIAL_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_DENSE_POLYNOMIAL_H_

#include "tachyon/c/export.h"

struct tachyon_bn254_univariate_dense_polynomial {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_dense_polynomial_create();

TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_dense_polynomial_clone(
    const tachyon_bn254_univariate_dense_polynomial* evals);

TACHYON_C_EXPORT void tachyon_bn254_univariate_dense_polynomial_destroy(
    tachyon_bn254_univariate_dense_polynomial* evals);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_DENSE_POLYNOMIAL_H_
