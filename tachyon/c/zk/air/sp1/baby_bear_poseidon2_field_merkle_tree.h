/**
 * @file baby_bear_poseidon2_field_merkle_tree.h
 * @brief Defines the interface for the field merkle tree used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"

struct tachyon_sp1_baby_bear_poseidon2_field_merkle_tree {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Clones an existing field merkle tree structure.
 *
 * Creates a deep copy of the given field merkle tree structure.
 *
 * @param tree A const pointer to the field merkle tree structure
 * to clone.
 * @return A pointer to the cloned field merkle tree structure.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_field_merkle_tree*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_clone(
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree);

/**
 * @brief Destroys a field merkle tree, freeing its resources.
 *
 * @param tree A pointer to the field merkle tree to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_destroy(
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree);

/**
 * @brief Serializes a field merkle tree to the byte array.
 *
 * @param tree A const pointer to the set of field merkle tree.
 * @param data A pointer to the byte array.
 * @param data_len A pointer to store the length of the byte array.
 */
TACHYON_C_EXPORT void
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_serialize(
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree,
    uint8_t* data, size_t* data_len);

/**
 * @brief Deserializes a field merkle tree from the byte array.
 *
 * @param data A const pointer to the byte array.
 * @param data_len The length of the byte array.
 * @return A pointer to the deserialized field merkle tree.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_field_merkle_tree*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_deserialize(
    const uint8_t* data, size_t data_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_H_
