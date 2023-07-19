#ifndef TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_

#include <stddef.h>
#include <stdio.h>

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/math/finite_fields/prime_field_mont_cuda.h"

namespace tachyon {
namespace math {
namespace kernels {

template <typename Config>
__global__ void Add(const PrimeFieldMontCuda<Config>* x,
                    const PrimeFieldMontCuda<Config>* y,
                    PrimeFieldMontCuda<Config>* result, size_t count) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = x[gid] + y[gid];
}

template <typename Config>
__global__ void Sub(const PrimeFieldMontCuda<Config>* x,
                    const PrimeFieldMontCuda<Config>* y,
                    PrimeFieldMontCuda<Config>* result, size_t count) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = x[gid] - y[gid];
}

}  // namespace kernels
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_
