#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

struct tachyon_halo2_bn254_argument_data {};

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT tachyon_halo2_bn254_argument_data*
tachyon_halo2_bn254_argument_data_create(size_t num_circuits);

TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_destroy(
    tachyon_halo2_bn254_argument_data* data);

TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_reserve_advice_columns(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_columns);

// Note that |column| is destroyed after this call.
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_advice_column(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_evaluations* column);

TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_reserve_advice_blinds(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_blinds);

TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_advice_blind(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    const tachyon_bn254_fr* value);

TACHYON_C_EXPORT void
tachyon_halo2_bn254_argument_data_reserve_instance_columns(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_columns);

// Note that |column| is destroyed after this call.
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_instance_column(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_evaluations* column);

TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_reserve_instance_polys(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    size_t num_polys);

// Note that |poly| is destroyed after this call.
TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_instance_poly(
    tachyon_halo2_bn254_argument_data* data, size_t circuit_idx,
    tachyon_bn254_univariate_dense_polynomial* poly);

TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_reserve_challenges(
    tachyon_halo2_bn254_argument_data* data, size_t num_challenges);

TACHYON_C_EXPORT void tachyon_halo2_bn254_argument_data_add_challenge(
    tachyon_halo2_bn254_argument_data* data, const tachyon_bn254_fr* value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_ARGUMENT_DATA_H_
