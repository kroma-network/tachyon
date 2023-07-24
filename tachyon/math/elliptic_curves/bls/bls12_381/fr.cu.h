#ifndef TACHYON_MATH_ELIIPTIC_CURVES_BLS_BLS12_381_FR_CU_H_
#define TACHYON_MATH_ELIIPTIC_CURVES_BLS_BLS12_381_FR_CU_H_

#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_mont_cuda.cu.h"
#endif  // TACHYON_CUDA

namespace tachyon {
namespace math {
namespace bls12_381 {

#if TACHYON_CUDA
using FrCuda = PrimeFieldMontCuda<FrConfig>;
#endif  // TACHYON_CUDA

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELIIPTIC_CURVES_BLS_BLS12_381_FR_CU_H_
