#ifndef TACHYON_MATH_FINITE_FIELDS_TEST_GF7_CUDA_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_TEST_GF7_CUDA_CU_H_

#include "tachyon/math/finite_fields/test/gf7.h"

#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_cuda.cu.h"
#endif  // TACHYON_CUDA

namespace tachyon::math {

#if TACHYON_CUDA
using GF7Cuda = PrimeFieldCuda<GF7Config>;
#endif  // TACHYON_CUDA

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_TEST_GF7_CUDA_CU_H_
