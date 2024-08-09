/**
 * @file bn254_univariate_dense_polynomial.h
 * @brief Univariate dense polynomial operations for the BN254 curve.
 *
 * This header file defines the structure and API for univariate dense
 * polynomials over the BN254 curve. This includes creation, cloning, and
 * destruction of dense polynomial structures, which are fundamental to various
 * cryptographic protocols and algorithms implemented on this curve.
 */

#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_DENSE_POLYNOMIAL_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_DENSE_POLYNOMIAL_H_

#include "tachyon/c/export.h"

/**
 * @struct tachyon_bn254_univariate_dense_polynomial
 * @brief Represents a univariate dense polynomial over the BN254 curve.
 *
 * This structure encapsulates a univariate dense polynomial, offering efficient
 * storage and manipulation for cryptographic computations.
 */
struct tachyon_bn254_univariate_dense_polynomial {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new instance of a univariate dense polynomial.
 *
 * Allocates and initializes a new univariate dense polynomial structure for use
 * in cryptographic algorithms.
 *
 * @return A pointer to the newly created polynomial structure.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_dense_polynomial_create();

/**
 * @brief Clones a univariate dense polynomial.
 *
 * Creates a deep copy of the given univariate dense polynomial.
 *
 * @param evals A pointer to the polynomial to be cloned.
 * @return A pointer to the cloned polynomial structure.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_dense_polynomial_clone(
    const tachyon_bn254_univariate_dense_polynomial* evals);

/**
 * @brief Destroys a univariate dense polynomial.
 *
 * Frees the memory allocated for a univariate dense polynomial structure,
 * effectively destroying it.
 *
 * @param evals A pointer to the polynomial to be destroyed.
 */
TACHYON_C_EXPORT void tachyon_bn254_univariate_dense_polynomial_destroy(
    tachyon_bn254_univariate_dense_polynomial* evals);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_DENSE_POLYNOMIAL_H_
