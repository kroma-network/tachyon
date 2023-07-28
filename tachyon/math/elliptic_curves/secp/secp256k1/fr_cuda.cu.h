#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FR_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FR_CU_H_

#include "tachyon/math/elliptic_curves/secp/secp256k1/fr.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_mont_cuda.cu.h"
#endif  // TACHYON_CUDA

namespace tachyon::math {
namespace secp256k1 {

#if TACHYON_CUDA
using FrCuda = PrimeFieldMontCuda<FrConfig>;
#endif  // TACHYON_CUDA

}  // namespace secp256k1
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FR_CU_H_
