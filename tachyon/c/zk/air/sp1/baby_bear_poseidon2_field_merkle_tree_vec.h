/**
 * @file baby_bear_poseidon2_field_merkle_tree_vec.h
 * @brief Defines the interface for the field merkle tree used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_VEC_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_VEC_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree.h"

struct tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new field merkle tree vector.
 *
 * @param rounds The number of rounds.
 * @return A pointer to the newly created field merkle tree vector.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_create(size_t rounds);

/**
 * @brief Clones an existing field merkle tree vector structure.
 *
 * Creates a deep copy of the given field merkle tree vector structure.
 *
 * @param tree_vec A const pointer to the field merkle tree vector structure
 * to clone.
 * @return A pointer to the cloned field merkle tree vector structure.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_clone(
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec);

/**
 * @brief Destroys a field merkle tree vector, freeing its resources.
 *
 * @param pcs A pointer to the field merkle tree vector to destroy.
 */
TACHYON_C_EXPORT void
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_destroy(
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec);

/**
 * @brief Sets field merkle tree.
 *
 * @param tree_vec A pointer to the field merkle tree vector.
 * @param round The round index of the point.
 * @param tree The const pointer to the field merkle tree.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_set(
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec,
    size_t round,
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_VEC_H_
