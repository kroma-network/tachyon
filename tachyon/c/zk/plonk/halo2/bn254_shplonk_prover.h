#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"
#include "tachyon/c/zk/base/bn254_blinder.h"
#include "tachyon/c/zk/plonk/halo2/bn254_argument_data.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

struct tachyon_halo2_bn254_shplonk_prover {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_halo2_bn254_shplonk_prover*
tachyon_halo2_bn254_shplonk_prover_create_from_unsafe_setup(
    uint8_t transcript_type, uint32_t k, const tachyon_bn254_fr* s);

TACHYON_C_EXPORT tachyon_halo2_bn254_shplonk_prover*
tachyon_halo2_bn254_shplonk_prover_create_from_params(uint8_t transcript_type,
                                                      uint32_t k,
                                                      const uint8_t* params,
                                                      size_t params_len);

TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_destroy(
    tachyon_halo2_bn254_shplonk_prover* prover);

TACHYON_C_EXPORT uint32_t tachyon_halo2_bn254_shplonk_prover_get_k(
    const tachyon_halo2_bn254_shplonk_prover* prover);

TACHYON_C_EXPORT size_t tachyon_halo2_bn254_shplonk_prover_get_n(
    const tachyon_halo2_bn254_shplonk_prover* prover);

TACHYON_C_EXPORT tachyon_bn254_blinder*
tachyon_halo2_bn254_shplonk_prover_get_blinder(
    tachyon_halo2_bn254_shplonk_prover* prover);

TACHYON_C_EXPORT const tachyon_bn254_univariate_evaluation_domain*
tachyon_halo2_bn254_shplonk_prover_get_domain(
    const tachyon_halo2_bn254_shplonk_prover* prover);

TACHYON_C_EXPORT tachyon_bn254_g1_jacobian*
tachyon_halo2_bn254_shplonk_prover_commit(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly);

TACHYON_C_EXPORT tachyon_bn254_g1_jacobian*
tachyon_halo2_bn254_shplonk_prover_commit_lagrange(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_univariate_evaluations* evals);

TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_set_rng_state(
    tachyon_halo2_bn254_shplonk_prover* prover, const uint8_t* state,
    size_t state_len);

TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_set_transcript_state(
    tachyon_halo2_bn254_shplonk_prover* prover, const uint8_t* state,
    size_t state_len);

TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_set_extended_domain(
    tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_plonk_proving_key* pk);

TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_create_proof(
    tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_plonk_proving_key* pk,
    tachyon_halo2_bn254_argument_data* data);

// If |proof| is NULL, then it populates |proof_len| with length to be used.
// If |proof| is not NULL, then it populates |proof| with the proof.
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_get_proof(
    const tachyon_halo2_bn254_shplonk_prover* prover, uint8_t* proof,
    size_t* proof_len);

TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_set_transcript_repr(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    tachyon_bn254_plonk_proving_key* pk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_H_
