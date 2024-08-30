/**
 * @file baby_bear_poseidon2_domains.h
 * @brief Defines the interface for the set of domains used within the
 * SP1(BabyBear + Poseidon2) proof system.
 */

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DOMAINS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DOMAINS_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"

struct tachyon_sp1_baby_bear_poseidon2_domains {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new set of domains.
 *
 * @param rounds The number of rounds.
 * @return A pointer to the newly created domains.
 */
TACHYON_C_EXPORT
tachyon_sp1_baby_bear_poseidon2_domains*
tachyon_sp1_baby_bear_poseidon2_domains_create(size_t rounds);

/**
 * @brief Destroys a set of domains, freeing its resources.
 *
 * @param domains A pointer to the domains to destroy.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_domains_destroy(
    tachyon_sp1_baby_bear_poseidon2_domains* domains);

/**
 * @brief Allocates memory for the set of domains.
 *
 * @param domains A pointer to the set of domains.
 * @param round The round.
 * @param size The size of the domains.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_domains_allocate(
    tachyon_sp1_baby_bear_poseidon2_domains* domains, size_t round,
    size_t size);

/**
 * @brief Sets up a domain.
 *
 * @param domains A pointer to the domains.
 * @param round The round index of the point.
 * @param idx The index of the domain.
 * @param log_n The logarithmic of domain size.
 * @param shift The shift value.
 */
TACHYON_C_EXPORT void tachyon_sp1_baby_bear_poseidon2_domains_set(
    tachyon_sp1_baby_bear_poseidon2_domains* domains, size_t round, size_t idx,
    uint32_t log_n, const tachyon_baby_bear* shift);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DOMAINS_H_
