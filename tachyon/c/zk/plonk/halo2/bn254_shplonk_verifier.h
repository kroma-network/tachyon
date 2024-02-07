#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

struct tachyon_halo2_bn254_shplonk_verifier {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_halo2_bn254_shplonk_verifier*
tachyon_halo2_bn254_shplonk_verifier_create_from_params(
    uint8_t transcript_type, uint32_t k, const uint8_t* params,
    size_t params_len, const uint8_t* proof, size_t proof_len);

TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_verifier_destroy(
    tachyon_halo2_bn254_shplonk_verifier* verifier);

// Note that |instance_columns_vec| is destroyed after this call.
TACHYON_C_EXPORT bool tachyon_halo2_bn254_shplonk_verifier_verify_proof(
    tachyon_halo2_bn254_shplonk_verifier* verifier,
    const tachyon_bn254_plonk_verifying_key* vkey,
    tachyon_halo2_bn254_instance_columns_vec* instance_columns_vec);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_H_
