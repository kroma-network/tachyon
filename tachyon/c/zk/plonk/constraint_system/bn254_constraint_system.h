#ifndef TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_H_
#define TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/zk/plonk/constraint_system/column_key.h"
#include "tachyon/c/zk/plonk/constraint_system/phase.h"

struct tachyon_bn254_plonk_constraint_system {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT uint32_t
tachyon_bn254_plonk_constraint_system_compute_blinding_factors(
    const tachyon_bn254_plonk_constraint_system* cs);

// If |phases| is NULL, then it populates |phases_len| with length to be used.
// If |phases| is not NULL, then it populates |phases| with advice column phases
// of |cs|.
TACHYON_C_EXPORT void
tachyon_bn254_plonk_constraint_system_get_advice_column_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len);

// If |phases| is NULL, then it populates |phases_len| with length to be used.
// If |phases| is not NULL, then it populates |phases| with challenge phases
// of |cs|.
TACHYON_C_EXPORT void
tachyon_bn254_plonk_constraint_system_get_challenge_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len);

// If |phases| is NULL, then it populates |phases_len| with length to be used.
// If |phases| is not NULL, then it populates |phases| with phases of |cs|.
TACHYON_C_EXPORT void tachyon_bn254_plonk_constraint_system_get_phases(
    const tachyon_bn254_plonk_constraint_system* cs, tachyon_phase* phases,
    size_t* phases_len);

TACHYON_C_EXPORT size_t
tachyon_bn254_plonk_constraint_system_get_num_fixed_columns(
    const tachyon_bn254_plonk_constraint_system* cs);

TACHYON_C_EXPORT size_t
tachyon_bn254_plonk_constraint_system_get_num_instance_columns(
    const tachyon_bn254_plonk_constraint_system* cs);

TACHYON_C_EXPORT size_t
tachyon_bn254_plonk_constraint_system_get_num_advice_columns(
    const tachyon_bn254_plonk_constraint_system* cs);

TACHYON_C_EXPORT size_t
tachyon_bn254_plonk_constraint_system_get_num_challenges(
    const tachyon_bn254_plonk_constraint_system* cs);

// If |constants| is NULL, then it populates |constants_len| with length to
// be used. If |constants| is not NULL, then it populates |constants| with
// constants of |cs|.
TACHYON_C_EXPORT void tachyon_bn254_plonk_constraint_system_get_constants(
    const tachyon_bn254_plonk_constraint_system* cs,
    tachyon_fixed_column_key* constants, size_t* constants_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_H_
