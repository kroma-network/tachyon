#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FR_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FR_CU_H_

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_cuda.cu.h"
#endif  // TACHYON_CUDA

namespace tachyon::math {
namespace bn254 {

#if TACHYON_CUDA
using FrCuda = PrimeFieldCuda<FrConfig>;
#endif  // TACHYON_CUDA

}  // namespace bn254
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FR_CU_H_
