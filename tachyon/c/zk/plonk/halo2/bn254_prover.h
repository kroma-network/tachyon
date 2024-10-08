/**
 * @file bn254_prover.h
 * @brief Interface for the Halo2 BN254 Prover.
 *
 * This header file defines the structure and API for the prover specific to
 * the Halo2 proof system on the BN254 curve. This includes functions for
 * creating provers, committing to polynomials or evaluations, and generating
 * proofs.
 */
#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_PROVER_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_PROVER_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"
#include "tachyon/c/zk/base/bn254_blinder.h"
#include "tachyon/c/zk/plonk/halo2/bn254_argument_data.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

/**
 * @struct tachyon_halo2_bn254_prover
 * @brief Structure representing a prover for the Halo2 protocol over BN254.
 *
 * Encapsulates the state and functionality required for constructing Halo2
 * proofs using the construction on the BN254 curve. This includes managing
 * commitments, handling randomness, and generating the cryptographic proof.
 */
struct tachyon_halo2_bn254_prover {
  uint8_t vendor;
  uint8_t pcs_type;
  void* extra;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a prover from unsafe setup parameters.
 *
 * @param vendor Identifier for the vendor.
 * @param pcs_type Identifier for the pcs type.
 * @param transcript_type Identifier for the transcript type.
 * @param k Security parameter.
 * @param s The setup parameter s as a scalar field element.
 * @return A const pointer to the newly created prover instance.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_prover*
tachyon_halo2_bn254_prover_create_from_unsafe_setup(uint8_t vendor,
                                                    uint8_t pcs_type,
                                                    uint8_t transcript_type,
                                                    uint32_t k,
                                                    const tachyon_bn254_fr* s);

/**
 * @brief Creates a prover from given parameters.
 *
 * Initializes a prover instance for the Halo2 protocol using the specified
 * parameters, facilitating the proof generation process for given circuits.
 *
 * @param vendor Identifier for the vendor.
 * @param pcs_type Identifier for the pcs type.
 * @param transcript_type The type of transcript to be used.
 * @param k The circuit size parameter.
 * @param params A const pointer to the parameters used for prover creation.
 * @param params_len The length of the parameters array.
 * @return A pointer to the newly created prover instance.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_prover*
tachyon_halo2_bn254_prover_create_from_params(uint8_t vendor, uint8_t pcs_type,
                                              uint8_t transcript_type,
                                              uint32_t k, const uint8_t* params,
                                              size_t params_len);

/**
 * @brief Destroys a prover instance, freeing its resources.
 *
 * @param prover A pointer to the prover to destroy.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_destroy(
    tachyon_halo2_bn254_prover* prover);

/**
 * @brief Retrieves the 'k' parameter of the prover.
 *
 * This function returns the circuit size parameter 'k', which is crucial for
 * the setup and execution of the Halo2 protocol.
 *
 * @param prover A const pointer to the prover.
 * @return The 'k' parameter of the prover.
 */
TACHYON_C_EXPORT uint32_t
tachyon_halo2_bn254_prover_get_k(const tachyon_halo2_bn254_prover* prover);

/**
 * @brief Retrieves the number of circuits handled by the prover.
 *
 * @param prover A const pointer to the prover.
 * @return The number of circuits that the prover is configured to handle.
 */
TACHYON_C_EXPORT size_t
tachyon_halo2_bn254_prover_get_n(const tachyon_halo2_bn254_prover* prover);

/**
 * @brief Retrieves the G2 affine representation of the setup parameter s.
 *
 * @param prover A const pointer to the prover.
 * @return A pointer to the G2 affine representation of s.
 */
TACHYON_C_EXPORT const tachyon_bn254_g2_affine*
tachyon_halo2_bn254_prover_get_s_g2(const tachyon_halo2_bn254_prover* prover);

/**
 * @brief Retrieves the blinder instance associated with the prover.
 *
 * @param prover A pointer to the prover.
 * @return A pointer to the blinder used by the prover.
 */
TACHYON_C_EXPORT tachyon_bn254_blinder* tachyon_halo2_bn254_prover_get_blinder(
    tachyon_halo2_bn254_prover* prover);

/**
 * @brief Retrieves the evaluation domain used by the prover.
 *
 * @param prover A const pointer to the prover.
 * @return A pointer to the evaluation domain used by the prover.
 */
TACHYON_C_EXPORT const tachyon_bn254_univariate_evaluation_domain*
tachyon_halo2_bn254_prover_get_domain(const tachyon_halo2_bn254_prover* prover);

/**
 * @brief Commits to a polynomial using the prover.
 *
 * This function generates a commitment to a given polynomial as part of the
 * proof generation process.
 *
 * @param prover A const pointer to the prover.
 * @param poly A const pointer to the polynomial to commit.
 * @return A pointer to the commitment in the form of a G1 projective point.
 */
TACHYON_C_EXPORT tachyon_bn254_g1_projective* tachyon_halo2_bn254_prover_commit(
    const tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly);

/**
 * @brief Commits to polynomial evaluations (Lagrange form) using the prover.
 *
 * This function generates a commitment to the evaluations of a polynomial,
 * allowing for efficient handling of the polynomial in its evaluated form.
 *
 * @param prover A const pointer to the prover.
 * @param evals A const pointer to the evaluations to commit.
 * @return A pointer to the commitment in the form of a G1 projective point.
 */
TACHYON_C_EXPORT tachyon_bn254_g1_projective*
tachyon_halo2_bn254_prover_commit_lagrange(
    const tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_univariate_evaluations* evals);

/**
 * @brief Marks the prover to prepare for batch commitment.
 *
 * @param prover A const pointer to the prover.
 * @param len The number of commitments.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_batch_start(
    const tachyon_halo2_bn254_prover* prover, size_t len);

/**
 * @brief Commits to a polynomial using the prover.
 *
 * Unlike \ref tachyon_halo2_bn254_prover_commit(), this function doesn't
 * generate a commitment immediately to avoid expensive inverse operations.
 *
 * @param prover A const pointer to the prover.
 * @param poly A const pointer to the polynomial to commit.
 * @param idx The index of the commitment.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_batch_commit(
    const tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly, size_t idx);

/**
 * @brief Commits to polynomial evaluations (Lagrange form) using the prover.
 *
 * Unlike \ref tachyon_halo2_bn254_prover_commit_lagrange(),
 * this function doesn't generate a commitment immediately to avoid expensive
 * inverse operation.
 *
 * @param prover A const pointer to the prover.
 * @param evals A const pointer to the evaluations to commit.
 * @param idx The index of the commitment.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_batch_commit_lagrange(
    const tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_univariate_evaluations* evals, size_t idx);

/**
 * @brief Retrieves the resulting commitment from the prover.
 *
 * @param prover A const pointer to the prover.
 * @param points A pointer to the affine points.
 * @param len The number of commitments.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_batch_end(
    const tachyon_halo2_bn254_prover* prover, tachyon_bn254_g1_affine* points,
    size_t len);

/**
 * @brief Sets the random number generator state for the prover.
 *
 * Configures the internal RNG state, ensuring the reproducibility and security
 * of the random numbers used in the proof generation process.
 *
 * @param prover A pointer to the prover.
 * @param rng_type Identifier for the rng type.
 * @param state A const pointer to the RNG state.
 * @param state_len Length of the RNG state array.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_set_rng_state(
    tachyon_halo2_bn254_prover* prover, uint8_t rng_type, const uint8_t* state,
    size_t state_len);

/**
 * @brief Sets the state of the transcript to a specific value.
 *
 * This is crucial for ensuring consistency and security across the entire proof
 * generation and verification process.
 *
 * @param prover A pointer to the prover.
 * @param state A const pointer to the transcript state.
 * @param state_len Length of the transcript state array.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_set_transcript_state(
    tachyon_halo2_bn254_prover* prover, const uint8_t* state, size_t state_len);

/**
 * @brief Sets the extended domain for the scroll versioned prover based on the
 * proving key.
 *
 * This function allows the prover to operate over an extended domain, which is
 * necessary for certain types of proofs that require a larger evaluation
 * domain.
 *
 * @param prover A pointer to the prover.
 * @param pk A const pointer to the proving key that contains the extended
 * domain.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_set_extended_domain(
    tachyon_halo2_bn254_prover* prover,
    const tachyon_bn254_plonk_proving_key* pk);

/**
 * @brief Initiates the proof creation process using the scroll versioned
 * prover, proving key, and argument data.
 *
 * @param prover A pointer to the prover.
 * @param pk A pointer to the proving key used for generating the proof.
 * @param data A pointer to the argument data required for the proof.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_create_proof(
    tachyon_halo2_bn254_prover* prover, tachyon_bn254_plonk_proving_key* pk,
    tachyon_halo2_bn254_argument_data* data);

/**
 * @brief Retrieves the generated proof.
 *
 * If the proof parameter is NULL, the function will provide the necessary
 * length for the proof via proof_len. If the proof parameter is not NULL, the
 * function will fill it with the generated proof data.
 *
 * @param prover A const pointer to the prover that generated the proof.
 * @param proof A pointer to the buffer where the proof will be stored.
 * @param proof_len A pointer to a variable where the length of the proof will
 * be stored.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_get_proof(
    const tachyon_halo2_bn254_prover* prover, uint8_t* proof,
    size_t* proof_len);

/**
 * @brief Sets the representation of the transcript for the scroll versioned
 * prover based on the proving key.
 *
 * This function configures how the prover should interpret the transcript,
 * impacting the format and structure of the generated proof.
 *
 * @param prover A const pointer to the prover.
 * @param pk A pointer to the proving key that dictates the transcript
 * representation.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_prover_set_transcript_repr(
    const tachyon_halo2_bn254_prover* prover,
    tachyon_bn254_plonk_proving_key* pk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_PROVER_H_
