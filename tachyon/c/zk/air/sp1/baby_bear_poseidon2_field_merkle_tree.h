/**
 * @file baby_bear_poseidon2_field_merkle_tree.h
 * @brief Defines the interface for the field merkle tree used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_H_

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
 * @param challenger A const pointer to the field merkle tree structure to
 * clone.
 * @return A pointer to the cloned field merkle tree structure.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_field_merkle_tree*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_clone(
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* challenger);

/**
 * @brief Destroys a field merkle tree, freeing its resources.
 *
 * @param pcs A pointer to the field merkle tree to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_destroy(
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* field_merkle_tree);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_H_
