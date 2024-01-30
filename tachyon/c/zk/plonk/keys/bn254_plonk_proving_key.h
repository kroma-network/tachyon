#ifndef TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_H_
#define TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

struct tachyon_bn254_plonk_proving_key {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_bn254_plonk_proving_key*
tachyon_bn254_plonk_proving_key_create_from_state(const uint8_t* state,
                                                  size_t state_len);

TACHYON_C_EXPORT void tachyon_bn254_plonk_proving_key_destroy(
    tachyon_bn254_plonk_proving_key* pk);

TACHYON_C_EXPORT const tachyon_bn254_plonk_verifying_key*
tachyon_bn254_plonk_proving_key_get_verifying_key(
    const tachyon_bn254_plonk_proving_key* pk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_H_
