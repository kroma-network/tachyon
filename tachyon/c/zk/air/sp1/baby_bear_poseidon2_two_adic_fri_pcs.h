/**
 * @file baby_bear_poseidon2_two_adic_fri_pcs.h
 * @brief Defines the interface for the two adic fri pcs used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/matrix/baby_bear_row_major_matrix.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_constants.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree.h"

struct tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new two adic fri pcs.
 *
 * @param log_blowup The logarithmic blowup factor.
 * @param num_queries The number of queries.
 * @param proof_of_work_bits The proof of work bits.
 * @return A pointer to the newly created two adic fri pcs.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs*
tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_create(
    uint32_t log_blowup, size_t num_queries, size_t proof_of_work_bits);

/**
 * @brief Destroys a two adic fri pcs, freeing its resources.
 *
 * @param pcs A pointer to the two adic fri pcs to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_destroy(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs* pcs);

/**
 * @brief Allocates internal memory required for storing low-degree extensions.
 *
 * @param pcs A pointer to the two adic fri pcs.
 * @param size The number of evaluations.
 */
TACHYON_C_EXPORT void
tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_allocate_ldes(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs* pcs, size_t size);

/**
 * @brief Compute the low-degree extension of each column of the matrix onto a
 * coset of a larger subgroup.
 *
 * @param pcs A pointer to the two adic fri pcs.
 * @param values A pointer to the data of the baby bear row major matrix.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @param shift The shift value.
 * @param new_rows The number of rows of the baby bear row major matrix.
 * @return A pointer to the data of the newly created baby bear row major
 * matrix.
 */
TACHYON_C_EXPORT tachyon_baby_bear*
tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_coset_lde_batch(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs* pcs,
    tachyon_baby_bear* values, size_t rows, size_t cols,
    tachyon_baby_bear shift, size_t* new_rows);

/**
 * @brief Commits to the mixed matrix created by
 * tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_coset_lde_batch.
 *
 * @param pcs A pointer to the two adic fri pcs.
 * @param commitment A pointer to store the commitment.
 * @param prover_data A pointer to store the field merkle tree.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_commit(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs* pcs,
    tachyon_baby_bear* commitment,
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree** prover_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_
