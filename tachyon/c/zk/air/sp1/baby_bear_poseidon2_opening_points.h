/**
 * @file baby_bear_poseidon2_opening_points.h
 * @brief Defines the interface for the set of opening points used within
 * the SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENING_POINTS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENING_POINTS_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear4.h"

struct tachyon_sp1_baby_bear_poseidon2_opening_points {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new set of opening points.
 *
 * @param rounds The number of rounds.
 * @return A pointer to the newly created set of opening points.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_opening_points*
tachyon_sp1_baby_bear_poseidon2_opening_points_create(size_t rounds);

/**
 * @brief Clones an existing set of opening points.
 *
 * Creates a deep copy of the given set of opening points.
 *
 * @param opening_points A const pointer to the set of opening points to
 * clone.
 * @return A pointer to the cloned set of opening points.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_opening_points*
tachyon_sp1_baby_bear_poseidon2_opening_points_clone(
    const tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points);

/**
 * @brief Destroys a set of opening points, freeing its resources.
 *
 * @param opening_points A pointer to the set of opening points to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_opening_points_destroy(
    tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points);

/**
 * @brief Allocates memory for the set of opening points.
 *
 * @param opening_points A pointer to the set of opening points.
 * @param round The round index of the point.
 * @param rows The number of rows.
 * @param cols The number of columns.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_opening_points_allocate(
    tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points,
    size_t round, size_t rows, size_t cols);

/**
 * @brief Sets point.
 *
 * @param opening_points A pointer to the set of opening points.
 * @param round The round index of the point.
 * @param row The row index of the point.
 * @param col The column index of the point.
 * @param col A const pointer to the point.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_opening_points_set(
    tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points,
    size_t round, size_t row, size_t col, const tachyon_baby_bear4* point);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENING_POINTS_H_
