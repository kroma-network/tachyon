/**
 * @file baby_bear_poseidon2_fri_proof.h
 * @brief Defines the interface for the fri proof used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FRI_PROOF_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FRI_PROOF_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"

struct tachyon_sp1_baby_bear_poseidon2_fri_proof {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new fri proof.
 *
 * @return A pointer to the newly created fri proof.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_fri_proof*
tachyon_sp1_baby_bear_poseidon2_fri_proof_create();

/**
 * @brief Clones an existing fri proof.
 *
 * Creates a deep copy of the given fri proof.
 *
 * @param fri_proof A const pointer to the fri proof to
 * clone.
 * @return A pointer to the cloned fri proof.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_fri_proof*
tachyon_sp1_baby_bear_poseidon2_fri_proof_clone(
    const tachyon_sp1_baby_bear_poseidon2_fri_proof* fri_proof);

/**
 * @brief Destroys a fri proof, freeing its resources.
 *
 * @param fri_proof A pointer to the fri proof to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_fri_proof_destroy(
    tachyon_sp1_baby_bear_poseidon2_fri_proof* fri_proof);

/**
 * @brief Writes a hint of the fri proof to the byte array.
 * See
 * https://github.com/succinctlabs/sp1/blob/6f67afd/crates/recursion/program/src/fri/hints.rs.
 *
 * @param fri_proof A const pointer to the set of fri proof.
 * @param data A pointer to the byte array.
 * @param data_len A pointer to store the length of the byte array.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_fri_proof_write_hint(
    const tachyon_sp1_baby_bear_poseidon2_fri_proof* fri_proof, uint8_t* data,
    size_t* data_len);

/**
 * @brief Serializes a fri proof to the byte array.
 *
 * @param fri_proof A const pointer to the set of fri proof.
 * @param data A pointer to the byte array.
 * @param data_len A pointer to store the length of the byte array.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_fri_proof_serialize(
    const tachyon_sp1_baby_bear_poseidon2_fri_proof* fri_proof, uint8_t* data,
    size_t* data_len);

/**
 * @brief Deserializes a fri proof from the byte array.
 *
 * @param data A const pointer to the byte array.
 * @param data_len The length of the byte array.
 * @return A pointer to the deserialized fri proof.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_fri_proof*
tachyon_sp1_baby_bear_poseidon2_fri_proof_deserialize(const uint8_t* data,
                                                      size_t data_len);

TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_fri_proof*
tachyon_sp1_baby_bear_poseidon2_fri_proof_deserialize_json(const uint8_t* data,
                                                           size_t data_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FRI_PROOF_H_
