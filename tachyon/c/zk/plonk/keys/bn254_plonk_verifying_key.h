/**
 * @file bn254_plonk_verifying_key.h
 * @brief This header file defines the verifying key structure for PLONK on the
 * BN254 curve.
 *
 * The verifying key is essential for verifying PLONK proofs.
 */
#ifndef TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_H_
#define TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system.h"

/**
 * @struct tachyon_bn254_plonk_verifying_key
 * @brief Represents the verifying key for PLONK protocol on the BN254 curve.
 *
 * Encapsulates data necessary for verifying PLONK proofs, including references
 * to the constraint system used in the proof and parameters for the evaluation
 * domain.
 */
struct tachyon_bn254_plonk_verifying_key {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Gets the constraint system associated with the verifying key.
 *
 * @param vk A const pointer to the verifying key.
 * @return A pointer to the constraint system used by the verifying key.
 */
TACHYON_C_EXPORT const tachyon_bn254_plonk_constraint_system*
tachyon_bn254_plonk_verifying_key_get_constraint_system(
    const tachyon_bn254_plonk_verifying_key* vk);

/**
 * @brief Gets the transcript representation used in the verifying key.
 *
 * @param vk A const pointer to the verifying key.
 * @return The transcript representation as a field element.
 */
TACHYON_C_EXPORT tachyon_bn254_fr
tachyon_bn254_plonk_verifying_key_get_transcript_repr(
    const tachyon_bn254_plonk_verifying_key* vk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_H_
