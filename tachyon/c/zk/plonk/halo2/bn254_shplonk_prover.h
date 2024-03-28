/**
 * @file
 * @brief SHPLONK Prover for Halo2 proofs over the BN254 curve.
 *
 * This component is responsible for creating SHPLONK proofs within the Halo2
 * proving system, leveraging the BN254 elliptic curve. It includes
 * functionality to initialize the prover, commit polynomials, and generate
 * proofs.
 */
#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_H_

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
 * @struct tachyon_halo2_bn254_shplonk_prover
 * @brief Represents a SHPLONK prover for Halo2 over the BN254 curve.
 *
 * Encapsulates the state and functionalities required to generate SHPLONK
 * proofs in the Halo2 framework, using BN254 as the underlying curve.
 */
struct tachyon_halo2_bn254_shplonk_prover {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a SHPLONK prover instance from an unsafe setup.
 *
 * @param transcript_type The type of transcript to use.
 * @param k Security parameter.
 * @param s The secret scalar.
 * @return A pointer to the newly created SHPLONK prover instance.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_shplonk_prover*
tachyon_halo2_bn254_shplonk_prover_create_from_unsafe_setup(
    uint8_t transcript_type, uint32_t k, const tachyon_bn254_fr* s);

/**
 * @brief Creates a SHPLONK prover instance from given parameters.
 *
 * @param transcript_type The type of transcript to use.
 * @param k Security parameter.
 * @param params Parameters for the prover setup.
 * @param params_len Length of the params array.
 * @return A pointer to the newly created SHPLONK prover instance.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_shplonk_prover*
tachyon_halo2_bn254_shplonk_prover_create_from_params(uint8_t transcript_type,
                                                      uint32_t k,
                                                      const uint8_t* params,
                                                      size_t params_len);

/**
 * @brief Destroys a SHPLONK prover instance, freeing associated resources.
 *
 * @param prover Pointer to the SHPLONK prover instance to destroy.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_destroy(
    tachyon_halo2_bn254_shplonk_prover* prover);

/**
 * @brief Retrieves the security parameter k used by the prover.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @return The security parameter k.
 */
TACHYON_C_EXPORT uint32_t tachyon_halo2_bn254_shplonk_prover_get_k(
    const tachyon_halo2_bn254_shplonk_prover* prover);

/**
 * @brief Retrieves the number of elements n in the evaluation domain.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @return The number of elements n in the evaluation domain.
 */
TACHYON_C_EXPORT size_t tachyon_halo2_bn254_shplonk_prover_get_n(
    const tachyon_halo2_bn254_shplonk_prover* prover);

/**
 * @brief Retrieves the G2 group generator scaled by the secret scalar s.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @return Pointer to the G2 group generator scaled by s.
 */
TACHYON_C_EXPORT const tachyon_bn254_g2_affine*
tachyon_halo2_bn254_shplonk_prover_get_s_g2(
    const tachyon_halo2_bn254_shplonk_prover* prover);

/**
 * @brief Retrieves the blinder instance used by the prover.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @return Pointer to the blinder instance.
 */
TACHYON_C_EXPORT tachyon_bn254_blinder*
tachyon_halo2_bn254_shplonk_prover_get_blinder(
    tachyon_halo2_bn254_shplonk_prover* prover);

/**
 * @brief Retrieves the evaluation domain used by the prover.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @return Pointer to the evaluation domain.
 */
TACHYON_C_EXPORT const tachyon_bn254_univariate_evaluation_domain*
tachyon_halo2_bn254_shplonk_prover_get_domain(
    const tachyon_halo2_bn254_shplonk_prover* prover);

/**
 * @brief Commits to a univariate dense polynomial by computing its evaluation
 * at the SRS (Structured Reference String) points.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @param poly Pointer to the univariate dense polynomial to commit to.
 * @return The commitment to the polynomial, represented as a point in G1.
 */
TACHYON_C_EXPORT tachyon_bn254_g1_jacobian*
tachyon_halo2_bn254_shplonk_prover_commit(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_univariate_dense_polynomial* poly);

/**
 * @brief Commits to evaluations of a polynomial in Lagrange form at the roots
 * of unity. This function is particularly useful for committing to polynomials
 * represented in the evaluation domain.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @param evals Pointer to the evaluations of the polynomial in Lagrange form.
 * @return The commitment to the evaluations, represented as a point in G1.
 */
TACHYON_C_EXPORT tachyon_bn254_g1_jacobian*
tachyon_halo2_bn254_shplonk_prover_commit_lagrange(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_univariate_evaluations* evals);

/**
 * @brief Sets the internal random number generator (RNG) state for the prover.
 * This is used for deterministic randomness in the proving process.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @param state The RNG state to set.
 * @param state_len The length of the RNG state.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_set_rng_state(
    tachyon_halo2_bn254_shplonk_prover* prover, const uint8_t* state,
    size_t state_len);

/**
 * @brief Sets the state of the transcript used by the prover. This is part of
 * the public-coin protocol and is necessary for verifiable random functions.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @param state The transcript state to set.
 * @param state_len The length of the transcript state.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_set_transcript_state(
    tachyon_halo2_bn254_shplonk_prover* prover, const uint8_t* state,
    size_t state_len);

/**
 * @brief Sets the extended evaluation domain based on the provided proving key.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @param pk Pointer to the proving key.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_set_extended_domain(
    tachyon_halo2_bn254_shplonk_prover* prover,
    const tachyon_bn254_plonk_proving_key* pk);

/**
 * @brief Generates a SHPLONK proof for the provided argument data.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @param pk Pointer to the proving key.
 * @param data Pointer to the argument data.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_create_proof(
    tachyon_halo2_bn254_shplonk_prover* prover,
    tachyon_bn254_plonk_proving_key* pk,
    tachyon_halo2_bn254_argument_data* data);

/**
 * @brief Retrieves the generated SHPLONK proof.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @param proof Buffer to store the proof.
 * @param proof_len Pointer to store the length of the proof.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_get_proof(
    const tachyon_halo2_bn254_shplonk_prover* prover, uint8_t* proof,
    size_t* proof_len);

/**
 * @brief Sets the representation of the transcript according to the proving
 * key. This is used for encoding the transcript in a specific way as defined by
 * the protocol.
 *
 * @param prover Pointer to the SHPLONK prover instance.
 * @param pk Pointer to the plonk proving key.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_shplonk_prover_set_transcript_repr(
    const tachyon_halo2_bn254_shplonk_prover* prover,
    tachyon_bn254_plonk_proving_key* pk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_H_
