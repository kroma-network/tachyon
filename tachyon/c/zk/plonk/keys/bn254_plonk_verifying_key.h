#ifndef TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_H_
#define TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system.h"

struct tachyon_bn254_plonk_verifying_key {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT const tachyon_bn254_plonk_constraint_system*
tachyon_bn254_plonk_verifying_key_get_constraint_system(
    const tachyon_bn254_plonk_verifying_key* vk);

TACHYON_C_EXPORT tachyon_bn254_fr
tachyon_bn254_plonk_verifying_key_get_transcript_repr(
    const tachyon_bn254_plonk_verifying_key* vk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_H_
