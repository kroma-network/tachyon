#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATIONS_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATIONS_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"

struct tachyon_bn254_univariate_evaluations {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluations_create();

TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluations_clone(
    const tachyon_bn254_univariate_evaluations* evals);

TACHYON_C_EXPORT void tachyon_bn254_univariate_evaluations_destroy(
    tachyon_bn254_univariate_evaluations* evals);

TACHYON_C_EXPORT size_t tachyon_bn254_univariate_evaluations_len(
    const tachyon_bn254_univariate_evaluations* evals);

TACHYON_C_EXPORT void tachyon_bn254_univariate_evaluations_set_value(
    tachyon_bn254_univariate_evaluations* evals, size_t i,
    const tachyon_bn254_fr* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATIONS_H_
