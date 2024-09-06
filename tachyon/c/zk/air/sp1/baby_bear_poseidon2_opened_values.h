/**
 * @file baby_bear_poseidon2_opened_values.h
 * @brief Defines the interface for the set of opened values used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENED_VALUES_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENED_VALUES_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear4.h"

struct tachyon_sp1_baby_bear_poseidon2_opened_values {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new set of opened values.
 *
 * @param rounds The number of rounds.
 * @return A pointer to the newly created set of opened values.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_opened_values*
tachyon_sp1_baby_bear_poseidon2_opened_values_create(size_t rounds);

/**
 * @brief Clones an existing set of opened values.
 *
 * Creates a deep copy of the given set of opened values.
 *
 * @param opened_values A const pointer to the set of opened values to
 * clone.
 * @return A pointer to the cloned set of opened values.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_opened_values*
tachyon_sp1_baby_bear_poseidon2_opened_values_clone(
    const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values);

/**
 * @brief Destroys a set of opened values, freeing its resources.
 *
 * @param opened_values A pointer to the set of opened values to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_opened_values_destroy(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values);

/**
 * @brief Serializes a set of opened values to the byte array.
 *
 * @param opened_values A const pointer to the set of opened values.
 * @param data A pointer to the byte array.
 * @param data_len A pointer to store the length of the byte array.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_opened_values_serialize(
    const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values,
    uint8_t* data, size_t* data_len);

/**
 * @brief Allocates outer memory for the set of opened values.
 *
 * @param opened_values A pointer to the set of opened values.
 * @param round The round index of the point.
 * @param rows The number of rows.
 * @param cols The number of columns.
 */
TACHYON_C_EXPORT void
tachyon_sp1_baby_bear_poseidon2_opened_values_allocate_outer(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values, size_t round,
    size_t rows, size_t cols);

/**
 * @brief Allocates inner memory for the set of opened values.
 *
 * @param opened_values A pointer to the set of opened values.
 * @param round The round index of the point.
 * @param row The row index of the value.
 * @param cols The number of columns.
 * @param size The size of the values.
 */
TACHYON_C_EXPORT void
tachyon_sp1_baby_bear_poseidon2_opened_values_allocate_inner(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values, size_t round,
    size_t row, size_t cols, size_t size);

/**
 * @brief Sets value.
 *
 * @param opened_values A pointer to the set of opened values.
 * @param round The round index of the point.
 * @param row The row index of the value.
 * @param col The column index of the value.
 * @param idx The index of the value.
 * @param value A const pointer to the value.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_opened_values_set(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values, size_t round,
    size_t row, size_t col, size_t idx, const tachyon_baby_bear4* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENED_VALUES_H_
