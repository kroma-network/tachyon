/**
 * @file
 * @brief Interface for the Halo2 BN254 Verifier.
 *
 * This header file defines the structure and API for the verifier specific
 * to the Halo2 proof system on the BN254 curve. It includes functionality for
 * creating verifiers from parameters, destroying verifiers, and verifying
 * proofs.
 */
#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_VERIFIER_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_VERIFIER_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

/**
 * @struct tachyon_halo2_bn254_verifier
 * @brief Structure representing a verifier for the Halo2 protocol on the BN254
 * curve.
 *
 * Encapsulates the functionality required for verifying Halo2 proofs. It is
 * responsible for checking the validity of proofs against a given verifying key
 * and instance data.
 */
struct tachyon_halo2_bn254_verifier {
  uint8_t vendor;
  uint8_t pcs_type;
  void* extra;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a verifier from provided parameters and a proof.
 *
 * Initializes a verifier instance for the Halo2 protocol using the specified
 * parameters and proof, facilitating the proof verification process.
 *
 * @param vendor The type of vendor used in the proof.
 * @param pcs_type The type of pcs used in the proof.
 * @param transcript_type The type of transcript used in the proof.
 * @param k The circuit size parameter.
 * @param params A pointer to the verifier parameters.
 * @param params_len The length of the parameters array.
 * @param proof A pointer to the proof to be verified.
 * @param proof_len The length of the proof.
 * @return A pointer to the newly created verifier instance.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_verifier*
tachyon_halo2_bn254_verifier_create_from_params(
    uint8_t vendor, uint8_t pcs_type, uint8_t transcript_type, uint32_t k,
    const uint8_t* params, size_t params_len, const uint8_t* proof,
    size_t proof_len);

/**
 * @brief Destroys a verifier instance, freeing its resources.
 *
 * @param verifier A pointer to the verifier to destroy.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_verifier_destroy(
    tachyon_halo2_bn254_verifier* verifier);

/**
 * @brief Verifies a proof against a given verifying key and instance columns.
 *
 * This function checks the validity of a proof with respect to the provided
 * verifying key and instance columns vector. Note that the instance columns
 * vector is destroyed after the call.
 *
 * @param verifier A pointer to the verifier.
 * @param vkey A pointer to the verifying key against which the proof will be
 * checked.
 * @param instance_columns_vec A pointer to the vector of instance columns
 * related to the proof.
 * @return True if the proof is valid, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_halo2_bn254_verifier_verify_proof(
    tachyon_halo2_bn254_verifier* verifier,
    const tachyon_bn254_plonk_verifying_key* vkey,
    tachyon_halo2_bn254_instance_columns_vec* instance_columns_vec);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_VERIFIER_H_
