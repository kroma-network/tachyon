#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_PRIME_FIELD_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_PRIME_FIELD_CU_H_

#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks_prime_field.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_mont_cuda.cu.h"
#endif  // TACHYON_CUDA

namespace tachyon {
namespace math {

#if TACHYON_CUDA
using GoldilocksCuda = PrimeFieldMontCuda<GoldilocksConfig>;
#endif  // TACHYON_CUDA

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_PRIME_FIELD_CU_H_
