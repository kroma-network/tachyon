/**
 * @file bn254_blinder.h
 * @brief Blinder for the BN254 Curve.
 *
 * This header file defines the structure and API for managing blinding factors
 * for the BN254 curve.
 */
#ifndef TACHYON_C_ZK_BASE_BN254_BLINDER_H_
#define TACHYON_C_ZK_BASE_BN254_BLINDER_H_

#include <stdint.h>

#include "tachyon/c/export.h"

/**
 * @struct tachyon_bn254_blinder
 * @brief Represents a blinder for BN254 curve.
 *
 * It allows setting and retrieving the blinding factors used
 * during the proof generation process.
 */
struct tachyon_bn254_blinder {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sets the blinding factors for a BN254 blinder.
 *
 * Configures the blinder with a specific number of blinding factors to be used
 * in zero-knowledge proof generation or verification.
 *
 * @param blinder Pointer to the BN254 blinder structure.
 * @param blinding_factors The number of blinding factors to be set.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_blinder_set_blinding_factors(
    tachyon_bn254_blinder* blinder, uint32_t blinding_factors);

/**
 * @brief Gets the blinding factors from a BN254 blinder.
 *
 * Retrieves the number of blinding factors configured in the blinder, which are
 * used in the zero-knowledge proof processes.
 *
 * @param blinder Pointer to the BN254 blinder structure.
 * @return The number of blinding factors set in the blinder.
 */
TACHYON_C_EXPORT uint32_t tachyon_halo2_bn254_blinder_get_blinding_factors(
    const tachyon_bn254_blinder* blinder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_BASE_BN254_BLINDER_H_
