/**
 * @file
 * @brief Univariate Rational Evaluations for BN254 Curve.
 *
 * This header file defines the structures and API for univariate rational
 * evaluations over the BN254 curve. It includes functionalities to create,
 * clone, and destroy rational evaluations structures, as well as setting values
 * within these structures and performing batch evaluations.
 */
#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_RATIONAL_EVALUATIONS_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_RATIONAL_EVALUATIONS_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

/**
 * @struct tachyon_bn254_univariate_rational_evaluations
 * @brief Represents a collection of univariate rational polynomial evaluations
 * over the BN254 curve.
 */
struct tachyon_bn254_univariate_rational_evaluations {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new univariate rational evaluations structure.
 *
 * Allocates and initializes a new structure for storing rational polynomial
 * evaluations.
 *
 * @return A pointer to the newly created rational evaluations structure.
 */

TACHYON_C_EXPORT tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_rational_evaluations_create();

/**
 * @brief Clones an existing univariate rational evaluations structure.
 *
 * Creates a deep copy of the given rational evaluations structure.
 *
 * @param evals A pointer to the rational evaluations structure to clone.
 * @return A pointer to the cloned rational evaluations structure.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_rational_evaluations_clone(
    const tachyon_bn254_univariate_rational_evaluations* evals);

/**
 * @brief Destroys a univariate rational evaluations structure.
 *
 * Frees the memory allocated for a rational evaluations structure.
 *
 * @param evals A pointer to the rational evaluations structure to destroy.
 */
TACHYON_C_EXPORT void tachyon_bn254_univariate_rational_evaluations_destroy(
    tachyon_bn254_univariate_rational_evaluations* evals);

/**
 * @brief Retrieves the length of the univariate rational evaluations structure.
 *
 * @param evals A pointer to the rational evaluations structure.
 * @return The number of rational evaluations stored in the structure.
 */
TACHYON_C_EXPORT size_t tachyon_bn254_univariate_rational_evaluations_len(
    const tachyon_bn254_univariate_rational_evaluations* evals);

/**
 * @brief Sets a specific index in the rational evaluations structure to zero.
 *
 * @param evals A pointer to the rational evaluations structure.
 * @param i Index at which to set the value to zero.
 */
TACHYON_C_EXPORT void tachyon_bn254_univariate_rational_evaluations_set_zero(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i);

/**
 * @brief Sets a specific index in the rational evaluations structure to a
 * trivial value (numerator only).
 *
 * @param evals A pointer to the rational evaluations structure.
 * @param i Index at which to set the value.
 * @param numerator A pointer to the numerator value.
 */
TACHYON_C_EXPORT void tachyon_bn254_univariate_rational_evaluations_set_trivial(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i,
    const tachyon_bn254_fr* numerator);

/**
 * @brief Sets a specific index in the rational evaluations structure to a
 * rational value (numerator and denominator).
 *
 * @param evals A pointer to the rational evaluations structure.
 * @param i Index at which to set the rational value.
 * @param numerator A pointer to the numerator value.
 * @param denominator A pointer to the denominator value.
 */
TACHYON_C_EXPORT void
tachyon_bn254_univariate_rational_evaluations_set_rational(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i,
    const tachyon_bn254_fr* numerator, const tachyon_bn254_fr* denominator);

/**
 * @brief Evaluates the rational evaluations structure at the given index.
 *
 * @param evals A pointer to the rational evaluations structure to evaluate.
 * @param i Index at which to perform the evaluation.
 * @param value A pointer to the value resulting from the evaluation.
 */
TACHYON_C_EXPORT void tachyon_bn254_univariate_rational_evaluations_evaluate(
    const tachyon_bn254_univariate_rational_evaluations* evals, size_t i,
    tachyon_bn254_fr* value);

/**
 * @brief Performs a batch evaluation on the rational evaluations structure.
 *
 * @param evals A pointer to the rational evaluations structure to evaluate.
 * @return A pointer to the univariate evaluations resulting from the batch
 * evaluation.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_rational_evaluations_batch_evaluate(
    const tachyon_bn254_univariate_rational_evaluations* evals);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_RATIONAL_EVALUATIONS_H_
