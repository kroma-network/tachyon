/**
 * @file
 * @brief Provides the interface for the SHPLONK verifier specific to the BN254
 * curve.
 */
#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

/**
 * @struct tachyon_halo2_bn254_shplonk_verifier
 * @brief Represents a SHPLONK verifier for Halo2 over the BN254 curve.
 */
struct tachyon_halo2_bn254_shplonk_verifier {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a SHPLONK verifier instance from provided parameters. This
 * verifier is capable of verifying SHPLONK proofs for the specified circuit
 * parameters and proof.
 *
 * @param transcript_type The type of transcript to be used.
 * @param k The circuit's security parameter.
 * @param params The serialized circuit parameters.
 * @param params_len The length of the serialized parameters.
 * @param proof The serialized proof to be verified.
 * @param proof_len The length of the serialized proof.
 * @return A pointer to the created SHPLONK verifier instance.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_shplonk_verifier*
tachyon_halo2_bn254_shplonk_verifier_create_from_params(
    uint8_t transcript_type, uint32_t k, const uint8_t* params,
    size_t params_len, const uint8_t* proof, size_t proof_len);

/**
 * @brief Destroys a SHPLONK verifier instance, freeing up any resources used.
 *
 * @param verifier A pointer to the SHPLONK verifier instance to be destroyed.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_verifier_destroy(
    tachyon_halo2_bn254_shplonk_verifier* verifier);

/**
 * @brief Verifies a SHPLONK proof using the provided verifying key and instance
 * columns vector. The instance columns vector provides the public inputs
 * to the proof.
 * Note: The instance_columns_vec is consumed and should not be used after
 * this call.
 *
 * @param verifier A pointer to the SHPLONK verifier instance.
 * @param vkey A pointer to the plonk verifying key used for the verification.
 * @param instance_columns_vec A pointer to the instance columns vector,
 * providing the public inputs necessary for the proof verification.
 * @return True if the proof is valid, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_halo2_bn254_shplonk_verifier_verify_proof(
    tachyon_halo2_bn254_shplonk_verifier* verifier,
    const tachyon_bn254_plonk_verifying_key* vkey,
    tachyon_halo2_bn254_instance_columns_vec* instance_columns_vec);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_H_
