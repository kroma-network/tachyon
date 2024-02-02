#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_RATIONAL_EVALUATIONS_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_RATIONAL_EVALUATIONS_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

struct tachyon_bn254_univariate_rational_evaluations {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_rational_evaluations_create();

TACHYON_C_EXPORT tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_rational_evaluations_clone(
    const tachyon_bn254_univariate_rational_evaluations* evals);

TACHYON_C_EXPORT void tachyon_bn254_univariate_rational_evaluations_destroy(
    tachyon_bn254_univariate_rational_evaluations* evals);

TACHYON_C_EXPORT size_t tachyon_bn254_univariate_rational_evaluations_len(
    const tachyon_bn254_univariate_rational_evaluations* evals);

TACHYON_C_EXPORT void tachyon_bn254_univariate_rational_evaluations_set_zero(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i);

TACHYON_C_EXPORT void tachyon_bn254_univariate_rational_evaluations_set_trivial(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i,
    const tachyon_bn254_fr* numerator);

TACHYON_C_EXPORT void
tachyon_bn254_univariate_rational_evaluations_set_rational(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i,
    const tachyon_bn254_fr* numerator, const tachyon_bn254_fr* denominator);

TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_rational_evaluations_batch_evaluate(
    const tachyon_bn254_univariate_rational_evaluations* evals);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_RATIONAL_EVALUATIONS_H_
