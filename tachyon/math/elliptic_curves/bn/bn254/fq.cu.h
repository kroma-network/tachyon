#ifndef TACHYON_MATH_ELIIPTIC_CURVES_BN_BN254_FQ_CU_H_
#define TACHYON_MATH_ELIIPTIC_CURVES_BN_BN254_FQ_CU_H_

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_mont_cuda.cu.h"
#endif  // TACHYON_CUDA

namespace tachyon {
namespace math {
namespace bn254 {

#if TACHYON_CUDA
using FqCuda = PrimeFieldMontCuda<FqConfig>;
#endif  // TACHYON_CUDA

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELIIPTIC_CURVES_BN_BN254_FQ_CU_H_
