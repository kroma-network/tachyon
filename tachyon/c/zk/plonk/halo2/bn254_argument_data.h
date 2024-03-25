/**
 * @file
 * @brief Halo2 Argument Data for BN254 Curve.
 *
 * This header file defines the structure and API for managing argument data in
 * Halo2 proofs over the BN254 curve. It includes functions for creating and
 * destroying argument data structures, reserving and adding advice columns,
 * instance columns, and challenges, as well as managing polynomial and blind
 * values associated with these columns.
 */
#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

/**
 * @struct tachyon_halo2_bn254_argument_data
 * @brief Represents argument data for a Halo2 proof on the BN254 curve.
 *
 * This structure encapsulates the data necessary for constructing and verifying
 * Halo2 proofs, including advice and instance columns, and challenges. It
 * allows for the efficient management and manipulation of this data as part of
 * the proof generation and verification process.
 */
struct tachyon_halo2_bn254_argument_data {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new Halo2 argument data structure.
 *
 * Allocates and initializes a new structure for storing Halo2 argument data
 * for a given number of circuits.
 *
 * @param num_circuits The number of circuits for which to create argument data.
 * @return Pointer to the newly created argument data structure.
 */
TACHYON_C_EXPORT tachyon_halo2_bn254_argument_data*
tachyon_halo2_bn254_argument_data_create(size_t num_circuits);

/**
 * @brief Destroys a Halo2 argument data structure.
 *
 * Frees the memory allocated for an argument data structure.
 *
 * @param data Pointer to the argument data structure to destroy.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_destroy(
    tachyon_halo2_bn254_argument_data* data);

/**
 * @brief Reserves space for a specified number of advice columns in a given
 * circuit.
 *
 * @param data Pointer to the argument data structure.
 * @param circuit_idx The index of the circuit for which to reserve advice
 * columns.
 * @param num_columns The number of advice columns to reserve.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_reserve_advice_columns(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_columns);

/**
 * @brief Adds an advice column to the argument data for a given circuit.
 * Note: The column object is consumed by this call and should not be used
 * afterwards.
 *
 * @param data Pointer to the argument data structure.
 * @param circuit_idx The index of the circuit to which the advice column is
 * added.
 * @param column Pointer to the advice column evaluations to add.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_advice_column(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_evaluations* column);

/**
 * @brief Reserves space for a specified number of advice blinds in a given
 * circuit.
 *
 * @param data Pointer to the argument data structure.
 * @param circuit_idx The index of the circuit for which to reserve advice
 * blinds.
 * @param num_blinds The number of advice blinds to reserve.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_reserve_advice_blinds(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_blinds);

/**
 * @brief Adds an advice blind value to the argument data for a given circuit.
 *
 * @param data Pointer to the argument data structure.
 * @param circuit_idx The index of the circuit to which the advice blind is
 * added.
 * @param value Pointer to the scalar field element representing the blind
 * value.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_advice_blind(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    const tachyon_bn254_fr* value);

/**
 * @brief Reserves space for a specified number of instance columns in a given
 * circuit.
 *
 * @param data Pointer to the argument data structure.
 * @param circuit_idx The index of the circuit for which to reserve instance
 * columns.
 * @param num_columns The number of instance columns to reserve.
 */
TACHYON_C_EXPORT void
tachyon_halo2_bn254_argument_data_reserve_instance_columns(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_columns);

/**
 * @brief Adds an instance column to the argument data for a given circuit.
 * Note: The column object is consumed by this call and should not be used
 * afterwards.
 *
 * @param data Pointer to the argument data structure.
 * @param circuit_idx The index of the circuit to which the instance column is
 * added.
 * @param column Pointer to the instance column evaluations to add.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_instance_column(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_evaluations* column);

/**
 * @brief Reserves space for a specified number of instance polynomials in a
 * given circuit.
 *
 * @param data Pointer to the argument data structure.
 * @param circuit_idx The index of the circuit for which to reserve instance
 * polynomials.
 * @param num_polys The number of instance polynomials to reserve.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_reserve_instance_polys(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_polys);

/**
 * @brief Adds an instance polynomial to the argument data for a given circuit.
 * Note: The poly object is consumed by this call and should not be used
 * afterwards.
 *
 * @param data Pointer to the argument data structure.
 * @param circuit_idx The index of the circuit to which the instance polynomial
 * is added.
 * @param poly Pointer to the instance polynomial to add.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_instance_poly(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_dense_polynomial* poly);

/**
 * @brief Reserves space for a specified number of challenges in the argument
 * data.
 *
 * This function allows for the reservation of space for challenge values that
 * will be used in the proof construction or verification. Preallocating space
 * can improve efficiency and manage memory usage effectively.
 *
 * @param data Pointer to the argument data structure.
 * @param num_challenges The number of challenges to reserve space for.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_reserve_challenges(
    tachyon_halo2_bn254_argument_data* data, size_t num_challenges);

/**
 * @brief Adds a challenge value to the argument data.
 *
 * This function appends a single challenge value to the argument data.
 * Challenges are essential elements in constructing and verifying Halo2 proofs,
 * providing randomness and contributing to the security of the protocol.
 *
 * @param data Pointer to the argument data structure.
 * @param value Pointer to the scalar field element representing the challenge
 * value.
 */
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_challenge(
    tachyon_halo2_bn254_argument_data* data, const tachyon_bn254_fr* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_H_
