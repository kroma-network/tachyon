#ifndef TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_

#include <stddef.h>
#include <stdio.h>

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/math/finite_fields/prime_field_mont_cuda.h"

namespace tachyon {
namespace math {
namespace kernels {

#define DEFINE_FIELD_OP(method, operator)                                    \
  template <typename Config>                                                 \
  __global__ void method(const PrimeFieldMontCuda<Config>* x,                \
                         const PrimeFieldMontCuda<Config>* y,                \
                         PrimeFieldMontCuda<Config>* result, size_t count) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (gid >= count) return;                                                \
    result[gid] = x[gid] operator y[gid];                                    \
  }

DEFINE_FIELD_OP(Add, +)
DEFINE_FIELD_OP(Sub, -)

#undef DEFINE_FIELD_OP

#define DEFINE_COMPARISON_OP(method, operator)                              \
  template <typename Config>                                                \
  __global__ void method(const PrimeFieldMontCuda<Config>* x,               \
                         const PrimeFieldMontCuda<Config>* y, bool* result, \
                         size_t count) {                                    \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (gid >= count) return;                                               \
    result[gid] = x[gid] operator y[gid];                                   \
  }

DEFINE_COMPARISON_OP(Eq, ==)
DEFINE_COMPARISON_OP(Ne, !=)
DEFINE_COMPARISON_OP(Lt, <)
DEFINE_COMPARISON_OP(Le, <=)
DEFINE_COMPARISON_OP(Gt, >)
DEFINE_COMPARISON_OP(Ge, >=)

#undef DEFINE_COMPARISON_OP

}  // namespace kernels
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_
