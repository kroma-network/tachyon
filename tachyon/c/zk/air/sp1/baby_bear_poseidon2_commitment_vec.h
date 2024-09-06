/**
 * @file baby_bear_poseidon2_commitment_vec.h
 * @brief Defines the interface for the commitment vector used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"

struct tachyon_sp1_baby_bear_poseidon2_commitment_vec {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new commitment vector.
 *
 * @param rounds The number of rounds.
 * @return A pointer to the newly created commitment vector.
 */
TACHYON_C_EXPORT
tachyon_sp1_baby_bear_poseidon2_commitment_vec*
tachyon_sp1_baby_bear_poseidon2_commitment_vec_create(size_t rounds);

/**
 * @brief Destroys a commitment vector, freeing its resources.
 *
 * @param commitment_vec A pointer to the commitment vector to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_commitment_vec_destroy(
    tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitment_vec);

/**
 * @brief Sets commitment.
 *
 * @param commitment_vec A pointer to the commitment vector.
 * @param round The round index of the point.
 * @param commitment A const pointer to the commitment.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_commitment_vec_set(
    tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitment_vec,
    size_t round, const tachyon_baby_bear* commitment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_H_
