/**
 * @file baby_bear_poseidon2_lde_vec.h
 * @brief Defines the interface for the lde vector used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_LDE_VEC_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_LDE_VEC_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"

struct tachyon_sp1_baby_bear_poseidon2_lde_vec {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new lde vector.
 *
 * @return A pointer to the newly created lde vector.
 */
TACHYON_C_EXPORT
tachyon_sp1_baby_bear_poseidon2_lde_vec*
tachyon_sp1_baby_bear_poseidon2_lde_vec_create();

/**
 * @brief Destroys a lde vector, freeing its resources.
 *
 * @param lde_vec A pointer to the lde vector to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_lde_vec_destroy(
    tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec);

/**
 * @brief Adds lde.
 *
 * @param lde_vec A pointer to the lde vector.
 * @param lde A const pointer to the lde.
 * @param rows The number of the rows.
 * @param cols The number of the cols.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_lde_vec_add(
    tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec,
    const tachyon_baby_bear* lde, size_t rows, size_t cols);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_LDE_VEC_H_
