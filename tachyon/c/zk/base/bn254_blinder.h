#ifndef TACHYON_C_ZK_BASE_BN254_BLINDER_H_
#define TACHYON_C_ZK_BASE_BN254_BLINDER_H_

#include <stdint.h>

#include "tachyon/c/export.h"

struct tachyon_bn254_blinder {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT void tachyon_halo2_bn254_blinder_set_blinding_factors(
    tachyon_bn254_blinder* blinder, uint32_t blinding_factors);

TACHYON_C_EXPORT uint32_t tachyon_halo2_bn254_blinder_get_blinding_factors(
    const tachyon_bn254_blinder* blinder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_BASE_BN254_BLINDER_H_
