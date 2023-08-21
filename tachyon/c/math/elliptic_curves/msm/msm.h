#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_H_

#include <stddef.h>

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/point2.h"

#ifdef __cplusplus
extern "C" {
#endif

TACHYON_C_EXPORT void tachyon_init_msm(uint8_t degree);

TACHYON_C_EXPORT void tachyon_release_msm();

TACHYON_C_EXPORT tachyon_bn254_g1_jacobian* tachyon_msm_g1_point2(
    const tachyon_bn254_point2* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len);

TACHYON_C_EXPORT tachyon_bn254_g1_jacobian* tachyon_msm_g1_affine(
    const tachyon_bn254_g1_affine* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_H_
