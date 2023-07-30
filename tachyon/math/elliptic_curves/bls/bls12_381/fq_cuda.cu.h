#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_CUDA_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_CUDA_CU_H_

#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_cuda.cu.h"
#endif  // TACHYON_CUDA

namespace tachyon::math {
namespace bls12_381 {

#if TACHYON_CUDA
using FqCuda = PrimeFieldCuda<FqConfig>;
#endif  // TACHYON_CUDA

}  // namespace bls12_381
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_CUDA_CU_H_
