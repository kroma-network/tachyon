/**
 * @file bn254_plonk_proving_key.h
 * @brief This header file defines the structure and functions for working with
 * the PLONK proving key for the BN254 curve.
 */
#ifndef TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_H_
#define TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

/**
 * @struct tachyon_bn254_plonk_proving_key
 * @brief Represents the proving key for a PLONK protocol on the BN254 curve.
 */
struct tachyon_bn254_plonk_proving_key {
  uint8_t ls_type;
  void* extra;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a PLONK proving key for the BN254 curve from a given state.
 *
 * @param ls_type Identifier for the ls type.
 * @param state A pointer to the buffer containing the serialized state of the
 * proving key.
 * @param state_len The length of the state buffer.
 * @return A pointer to the newly created PLONK proving key.
 */
TACHYON_C_EXPORT tachyon_bn254_plonk_proving_key*
tachyon_bn254_plonk_proving_key_create_from_state(uint8_t ls_type,
                                                  const uint8_t* state,
                                                  size_t state_len);

/**
 * @brief Destroys a PLONK proving key for the BN254 curve, freeing its
 * resources.
 *
 * @param pk A pointer to the PLONK proving key to destroy.
 */
TACHYON_C_EXPORT void tachyon_bn254_plonk_proving_key_destroy(
    tachyon_bn254_plonk_proving_key* pk);

/**
 * @brief Retrieves the corresponding verifying key for a given PLONK proving
 * key.
 *
 * @param pk A pointer to the PLONK proving key.
 * @return A pointer to the corresponding PLONK verifying key.
 */
TACHYON_C_EXPORT const tachyon_bn254_plonk_verifying_key*
tachyon_bn254_plonk_proving_key_get_verifying_key(
    const tachyon_bn254_plonk_proving_key* pk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_H_
