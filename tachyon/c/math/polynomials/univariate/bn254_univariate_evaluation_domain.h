/**
 * @file
 * @brief Univariate Evaluation Domain for BN254 Curve.
 *
 * This header file defines the univariate evaluation domain structure and
 * related operations for the BN254 curve. It includes functions for creating
 * and destroying evaluation domains, generating empty polynomial and evaluation
 * structures, and performing Fast Fourier Transforms (FFT) and inverse FFT
 * (IFFT) on polynomials.
 */
#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_rational_evaluations.h"

/**
 * @struct tachyon_bn254_univariate_evaluation_domain
 * @brief Represents an evaluation domain for univariate polynomials over the
 * BN254 curve.
 */
struct tachyon_bn254_univariate_evaluation_domain {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new evaluation domain for a given number of coefficients.
 *
 * @param num_coeffs The number of coefficients in the domain.
 * @return A pointer to the newly created evaluation domain.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_evaluation_domain*
tachyon_bn254_univariate_evaluation_domain_create(size_t num_coeffs);

/**
 * @brief Destroys an evaluation domain, freeing its allocated resources.
 *
 * @param domain A pointer to the evaluation domain to destroy.
 */
TACHYON_C_EXPORT void tachyon_bn254_univariate_evaluation_domain_destroy(
    tachyon_bn254_univariate_evaluation_domain* domain);

/**
 * @brief Creates an empty evaluations structure associated with the evaluation
 * domain.
 *
 * @param domain A pointer to the evaluation domain.
 * @return A pointer to the empty evaluations structure.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluation_domain_empty_evals(
    const tachyon_bn254_univariate_evaluation_domain* domain);

/**
 * @brief Creates an empty dense polynomial associated with the evaluation
 * domain.
 *
 * @param domain A pointer to the evaluation domain.
 * @return A pointer to the empty dense polynomial.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_evaluation_domain_empty_poly(
    const tachyon_bn254_univariate_evaluation_domain* domain);

/**
 * @brief Creates an empty rational evaluations structure associated with the
 * evaluation domain.
 *
 * @param domain A pointer to the evaluation domain.
 * @return A pointer to the empty rational evaluations structure.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_evaluation_domain_empty_rational_evals(
    const tachyon_bn254_univariate_evaluation_domain* domain);

/**
 * @brief Performs the Fast Fourier Transform (FFT) on a given polynomial within
 * the domain.
 *
 * @param domain A pointer to the evaluation domain.
 * @param poly A pointer to the polynomial to transform.
 * @return A pointer to the evaluations resulting from the FFT.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluation_domain_fft(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    const tachyon_bn254_univariate_dense_polynomial* poly);

/**
 * @brief Performs the in-place Fast Fourier Transform (FFT) on a given
 * polynomial within the domain. Note that memory space in poly is altered
 * after this call.
 *
 * @param domain A pointer to the evaluation domain.
 * @param poly A pointer to the polynomial to transform.
 * @return A pointer to the evaluations resulting from the FFT.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluation_domain_fft_inplace(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    tachyon_bn254_univariate_dense_polynomial* poly);

/**
 * @brief Performs the inverse Fast Fourier Transform (IFFT) on given
 * evaluations within the domain.
 *
 * @param domain A pointer to the evaluation domain.
 * @param evals A pointer to the evaluations to transform back into a
 * polynomial.
 * @return A pointer to the dense polynomial resulting from the IFFT.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_evaluation_domain_ifft(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    const tachyon_bn254_univariate_evaluations* evals);

/**
 * @brief Performs the in-place inverse Fast Fourier Transform (IFFT) on given
 * evaluations within the domain. Note that memory space in evals is altered
 * after this call.
 *
 * @param domain A pointer to the evaluation domain.
 * @param evals A pointer to the evaluations to transform back into a
 * polynomial.
 * @return A pointer to the dense polynomial resulting from the IFFT.
 */
TACHYON_C_EXPORT tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_evaluation_domain_ifft_inplace(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    tachyon_bn254_univariate_evaluations* evals);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_H_
