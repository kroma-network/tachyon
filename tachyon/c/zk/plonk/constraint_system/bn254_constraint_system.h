/**
 * @file bn254_constraint_system.h
 * @brief PLONK Constraint System for BN254 Curve.
 *
 * This header file defines the API for the PLONK constraint system specific to
 * the BN254 curve, offering functionalities to compute blinding factors,
 * retrieve column phases, and access counts of fixed, instance, and advice
 * columns as well as challenges and constants within the constraint system.
 */
#ifndef TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_H_
#define TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/plonk/constraint_system/column_key.h"
#include "tachyon/c/zk/plonk/constraint_system/phase.h"

/**
 * @struct tachyon_bn254_plonk_constraint_system
 * @brief Represents the PLONK constraint system structure for the BN254 curve.
 */
struct tachyon_bn254_plonk_constraint_system {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the blinding factors necessary for the constraint system.
 *
 * @param cs Pointer to the constraint system structure.
 * @return The number of blinding factors computed for the constraint system.
 */
TACHYON_C_EXPORT uint32_t
tachyon_bn254_plonk_constraint_system_compute_blinding_factors(
    const tachyon_bn254_plonk_constraint_system* cs);

/**
 * @brief Retrieves the phases for advice columns within the constraint system.
 *
 * @param cs Pointer to the constraint system structure.
 * @param phases Output array to store the phases. Can be NULL to query required
 * length.
 * @param phases_len Pointer to store the length of phases array or required
 * length if phases is NULL.
 */
TACHYON_C_EXPORT void
tachyon_bn254_plonk_constraint_system_get_advice_column_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len);

/**
 * @brief Retrieves the phases for challenge columns within the constraint
 * system.
 *
 * @param cs Pointer to the constraint system structure.
 * @param phases Output array to store the phases. Can be NULL to query required
 * length.
 * @param phases_len Pointer to store the length of phases array or required
 * length if phases is NULL.
 */
TACHYON_C_EXPORT void
tachyon_bn254_plonk_constraint_system_get_challenge_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len);

/**
 * @brief Retrieves all phases within the constraint system.
 *
 * @param cs Pointer to the constraint system structure.
 * @param phases Output array to store the phases. Can be NULL to query required
 * length.
 * @param phases_len Pointer to store the length of phases array or required
 * length if phases is NULL.
 */
TACHYON_C_EXPORT void tachyon_bn254_plonk_constraint_system_get_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len);

/**
 * @brief Retrieves the number of fixed columns in the constraint system.
 *
 * @param cs Pointer to the constraint system structure.
 * @return The number of fixed columns.
 */
TACHYON_C_EXPORT size_t
tachyon_bn254_plonk_constraint_system_get_num_fixed_columns(
    const tachyon_bn254_plonk_constraint_system* cs);

/**
 * @brief Retrieves the number of instance columns in the constraint system.
 *
 * @param cs Pointer to the constraint system structure.
 * @return The number of instance columns.
 */
TACHYON_C_EXPORT size_t
tachyon_bn254_plonk_constraint_system_get_num_instance_columns(
    const tachyon_bn254_plonk_constraint_system* cs);

/**
 * @brief Retrieves the number of advice columns in the constraint system.
 *
 * @param cs Pointer to the constraint system structure.
 * @return The number of advice columns.
 */
TACHYON_C_EXPORT size_t
tachyon_bn254_plonk_constraint_system_get_num_advice_columns(
    const tachyon_bn254_plonk_constraint_system* cs);

/**
 * @brief Retrieves the number of challenges in the constraint system.
 *
 * @param cs Pointer to the constraint system structure.
 * @return The number of challenges.
 */
TACHYON_C_EXPORT size_t
tachyon_bn254_plonk_constraint_system_get_num_challenges(
    const tachyon_bn254_plonk_constraint_system* cs);

/**
 * @brief Retrieves the constants used in the constraint system. This function
 * can be used in two modes: querying the required length of the constants array
 * or populating the constants array.
 *
 * If the `constants` parameter is NULL, this function will populate
 * `constants_len` with the required length for the constants array. This mode
 * allows the caller to determine the size of the array needed to hold all
 * constants.
 *
 * If the `constants` parameter is not NULL, the function assumes that
 * `constants` points to an array of sufficient size to hold all constants. It
 * will then populate this array with the constants from the constraint system.
 *
 * @param cs Pointer to the constraint system structure from which to retrieve
 * constants.
 * @param constants Pointer to the array where the constants will be stored. If
 * NULL, the function will only populate `constants_len` with the required array
 * size.
 * @param constants_len Pointer to a size_t variable where the function will
 * store the length of the constants array. If `constants` is NULL, it stores
 * the required length; otherwise, it stores the number of constants actually
 * populated.
 */
TACHYON_C_EXPORT void tachyon_bn254_plonk_constraint_system_get_constants(
    const tachyon_bn254_plonk_constraint_system* cs,
    tachyon_fixed_column_key* constants, size_t* constants_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_H_
