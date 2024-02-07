#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"

struct tachyon_halo2_bn254_instance_columns_vec {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_halo2_bn254_instance_columns_vec*
tachyon_halo2_bn254_instance_columns_vec_create(size_t num_circuits);

TACHYON_C_EXPORT void tachyon_halo2_bn254_instance_columns_vec_destroy(
    tachyon_halo2_bn254_instance_columns_vec* data);

TACHYON_C_EXPORT void tachyon_halo2_bn254_instance_columns_vec_resize_columns(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t num_columns);

TACHYON_C_EXPORT void tachyon_halo2_bn254_instance_columns_vec_reserve_values(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t column_idx, size_t num_values);

TACHYON_C_EXPORT void tachyon_halo2_bn254_instance_columns_vec_add_values(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t column_idx, const tachyon_bn254_fr* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_H_
