/**
 * @file baby_bear_poseidon2_duplex_challenger.h
 * @brief Defines the interface for the duplex challenger used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_H_

#include "tachyon/c/export.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"

struct tachyon_sp1_baby_bear_poseidon2_duplex_challenger {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new duplex challenger.
 *
 * @return A pointer to the newly created duplex challenger.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_duplex_challenger*
tachyon_sp1_baby_bear_poseidon2_duplex_challenger_create();

/**
 * @brief Clones an existing duplex challenger structure.
 *
 * Creates a deep copy of the given duplex challenger structure.
 *
 * @param challenger A const pointer to the duplex challenger structure to
 * clone.
 * @return A pointer to the cloned duplex challenger structure.
 */
TACHYON_C_EXPORT tachyon_sp1_baby_bear_poseidon2_duplex_challenger*
tachyon_sp1_baby_bear_poseidon2_duplex_challenger_clone(
    const tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger);

/**
 * @brief Destroys a duplex challenger, freeing its resources.
 *
 * @param challenger A pointer to the duplex challenger to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_duplex_challenger_destroy(
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger);

/**
 * @brief Observes a new value.
 *
 * @param challenger A pointer to the duplex challenger.
 * @param value A const pointer to the value.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_duplex_challenger_observe(
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger,
    const tachyon_baby_bear* value);

/**
 * @brief Samples a field element from the challenger.
 *
 * @param challenger A pointer to the duplex challenger.
 */
TACHYON_C_EXPORT tachyon_baby_bear
tachyon_sp1_baby_bear_poseidon2_duplex_challenger_sample(
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_H_
