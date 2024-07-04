/**
 * @file
 * @brief Univariate Evaluations for BN254 Curve.
 *
 * This header file defines the structure and API for managing univariate
 * evaluations over the BN254 curve. It includes functions for creating,
 * cloning, destroying, and manipulating evaluations, as well as setting and
 * retrieving values within these structures.
 */
#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATIONS_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATIONS_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"

/**
 * @struct tachyon_bn254_univariate_evaluations
 * @brief Represents a collection of univariate polynomial evaluations over the
 * BN254 curve.
 */
struct tachyon_bn254_univariate_evaluations {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new univariate evaluations structure.
 *
 * Allocates and initializes a new structure for storing polynomial evaluations.
 *
 * @return A pointer to the newly created evaluations structure.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluations_create();

/**
 * @brief Clones an existing univariate evaluations structure.
 *
 * Creates a deep copy of the given evaluations structure.
 *
 * @param evals A pointer to the evaluations structure to clone.
 * @return A pointer to the cloned evaluations structure.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluations_clone(
    const tachyon_bn254_univariate_evaluations* evals);

/**
 * @brief Destroys a univariate evaluations structure.
 *
 * Frees the memory allocated for an evaluations structure.
 *
 * @param evals A pointer to the evaluations structure to destroy.
 */
TACHYON_C_EXPORT void tachyon_bn254_univariate_evaluations_destroy(
    tachyon_bn254_univariate_evaluations* evals);

/**
 * @brief Retrieves the length of the univariate evaluations structure.
 *
 * @param evals A pointer to the evaluations structure.
 * @return The number of evaluations stored in the structure.
 */
TACHYON_C_EXPORT size_t tachyon_bn254_univariate_evaluations_len(
    const tachyon_bn254_univariate_evaluations* evals);

/**
 * @brief Sets a value in the univariate evaluations structure.
 *
 * @param evals A pointer to the evaluations structure.
 * @param i Index at which to set the value.
 * @param value A pointer to the value to set at index i.
 */
TACHYON_C_EXPORT void tachyon_bn254_univariate_evaluations_set_value(
    tachyon_bn254_univariate_evaluations* evals, size_t i,
    const tachyon_bn254_fr* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATIONS_H_
