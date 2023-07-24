#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CU_H_

#include "tachyon/math/finite_fields/prime_field.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_mont_cuda.h"
#endif  // TACHYON_CUDA

namespace tachyon {
namespace math {

#if TACHYON_CUDA
using GF7Cuda = PrimeFieldMontCuda<GF7Config>;
#endif  // TACHYON_CUDA

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CU_H_
