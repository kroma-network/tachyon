#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_CUDA_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_CUDA_CU_H_

#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks_cuda.cu.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_cuda.cu.h"
#endif  // TACHYON_CUDA

namespace tachyon::math {

#if TACHYON_CUDA
using GoldilocksCuda = PrimeFieldCuda<GoldilocksConfig>;
#endif  // TACHYON_CUDA

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_CUDA_CU_H_
