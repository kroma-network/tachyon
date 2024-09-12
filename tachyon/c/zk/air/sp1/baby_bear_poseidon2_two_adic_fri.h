/**
 * @file baby_bear_poseidon2_two_adic_fri.h
 * @brief Defines the interface for the two adic fri used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/c/math/matrix/baby_bear_row_major_matrix.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_commitment_vec.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_constants.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_domains.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_lde_vec.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opening_points.h"

struct tachyon_sp1_baby_bear_poseidon2_two_adic_fri {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new two adic fri.
 *
 * @param log_blowup The logarithmic blowup factor.
 * @param num_queries The number of queries.
 * @param proof_of_work_bits The proof of work bits.
 * @return A pointer to the newly created two adic fri.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_two_adic_fri*
tachyon_sp1_baby_bear_poseidon2_two_adic_fri_create(uint32_t log_blowup,
                                                    size_t num_queries,
                                                    size_t proof_of_work_bits);

/**
 * @brief Destroys a two adic fri, freeing its resources.
 *
 * @param pcs A pointer to the two adic fri to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_destroy(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs);

/**
 * @brief Compute the low-degree extension of each column of the matrix onto a
 * coset of a larger subgroup.
 *
 * @param pcs A pointer to the two adic fri.
 * @param values A pointer to the data of the baby bear row major matrix.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @param extended_values A pointer to the data of the extended baby bear row
 * major matrix.
 * @param shift The shift value.
 */
TACHYON_C_EXPORT void
tachyon_sp1_baby_bear_poseidon2_two_adic_fri_coset_lde_batch(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs,
    tachyon_baby_bear* values, size_t rows, size_t cols,
    tachyon_baby_bear* extended_values, tachyon_baby_bear shift);

/**
 * @brief Commits to the lde vector.
 *
 * @param pcs A pointer to the two adic fri.
 * @param lde_vec A pointer to the lde vector.
 * @param commitment A pointer to store the commitment.
 * @param prover_data A pointer to store the field merkle tree.
 * @param prover_data_vec A pointer to the field merkle tree vector.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_commit(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs,
    tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec,
    tachyon_baby_bear* commitment,
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree** prover_data,
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* prover_data_vec);

/**
 * @brief Creates an opening proof with prover data and points.
 *
 * @param pcs A const pointer to the two adic fri pcs.
 * @param prover_data_by_round A const pointer to the field merkle tree vector.
 * @param points_by_round A const pointer to the opening points.
 * @param challenger A pointer to the duplex challenger.
 * @param opened_values A pointer to store the opened values.
 * @param proof A pointer to store the fri proof.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_open(
    const tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs,
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec*
        prover_data_by_round,
    const tachyon_sp1_baby_bear_poseidon2_opening_points* points_by_round,
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger,
    tachyon_sp1_baby_bear_poseidon2_opened_values** opened_values,
    tachyon_sp1_baby_bear_poseidon2_fri_proof** proof);

/**
 * @brief Verifies an opening proof with commitments.
 *
 * @param pcs A const pointer to the two adic fri pcs.
 * @param commitments_by_round A const pointer to the commitment vector.
 * @param domains_by_round A const pointer to the domains.
 * @param prover_data_by_round A const pointer to the field merkle tree vector.
 * @param points_by_round A const pointer to the opening points.
 * @param opened_values_by_round A const pointer to the opened values.
 * @param proof A const pointer to the fri proof.
 * @param challenger A pointer to the duplex challenger.
 */
TACHYON_C_EXPORT bool tachyon_sp1_baby_bear_poseidon2_two_adic_fri_verify(
    const tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs,
    const tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitments_by_round,
    const tachyon_sp1_baby_bear_poseidon2_domains* domains_by_round,
    const tachyon_sp1_baby_bear_poseidon2_opening_points* points_by_round,
    const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values_by_round,
    const tachyon_sp1_baby_bear_poseidon2_fri_proof* proof,
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_H_
