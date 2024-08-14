/**
 * @file bn254_instance_columns_vec.h
 * @brief Defines the vector of instance columns for Halo2 proofs on the BN254
 * curve.
 *
 * This structure is used to manage instance columns for circuits, providing
 * functionality for creating, resizing, and managing the values within each
 * column of the instance data.
 */
#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"

/**
 * @struct tachyon_halo2_bn254_instance_columns_vec
 * @brief Represents a vector of instance columns for the Halo2 protocol.
 *
 * Encapsulates the instance data for one or more circuits, with functionality
 * to manipulate the size and content of the instance columns required for proof
 * verification.
 */
struct tachyon_halo2_bn254_instance_columns_vec {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new instance columns vector for a specified number of
 * circuits.
 *
 * @param num_circuits The number of circuits for which to create instance
 * columns.
 * @return A pointer to the newly created instance columns vector.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_instance_columns_vec*
tachyon_halo2_bn254_instance_columns_vec_create(size_t num_circuits);

/**
 * @brief Destroys an instance columns vector, freeing its resources.
 *
 * @param data A pointer to the instance columns vector to destroy.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_instance_columns_vec_destroy(
    tachyon_halo2_bn254_instance_columns_vec* data);

/**
 * @brief Resizes the number of columns for a specific circuit within the
 * vector.
 *
 * @param data A pointer to the instance columns vector.
 * @param circuit_idx Index of the circuit to resize.
 * @param num_columns The new number of columns for the specified circuit.
 */

TACHYON_C_EXPORT void tachyon_halo2_bn254_instance_columns_vec_resize_columns(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t num_columns);

/**
 * @brief Reserves space for values in a specific column of a circuit.
 *
 * @param data A pointer to the instance columns vector.
 * @param circuit_idx Index of the circuit containing the column.
 * @param column_idx Index of the column to reserve space in.
 * @param num_values The number of values to reserve space for.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_instance_columns_vec_reserve_values(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t column_idx, size_t num_values);

/**
 * @brief Adds values to a specific column of a circuit.
 *
 * @param data A pointer to the instance columns vector.
 * @param circuit_idx Index of the circuit containing the column.
 * @param column_idx Index of the column to add values to.
 * @param value A const pointer to the values to add.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_instance_columns_vec_add_values(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t column_idx, const tachyon_bn254_fr* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_H_
