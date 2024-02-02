#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_rational_evaluations.h"

struct tachyon_bn254_univariate_evaluation_domain {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_bn254_univariate_evaluation_domain*
tachyon_bn254_univariate_evaluation_domain_create(size_t num_coeffs);

TACHYON_C_EXPORT void tachyon_bn254_univariate_evaluation_domain_destroy(
    tachyon_bn254_univariate_evaluation_domain* domain);

TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluation_domain_empty_evals(
    const tachyon_bn254_univariate_evaluation_domain* domain);

TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_evaluation_domain_empty_poly(
    const tachyon_bn254_univariate_evaluation_domain* domain);

TACHYON_C_EXPORT tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_evaluation_domain_empty_rational_evals(
    const tachyon_bn254_univariate_evaluation_domain* domain);

TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluation_domain_fft(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    const tachyon_bn254_univariate_dense_polynomial* poly);

TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_evaluation_domain_ifft(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    const tachyon_bn254_univariate_evaluations* evals);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_H_
