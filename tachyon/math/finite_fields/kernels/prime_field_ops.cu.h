#ifndef TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_
#define TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_

#include <stddef.h>

#include "third_party/gpus/cuda/include/cuda_runtime.h"

namespace tachyon::math::kernels {

#define DEFINE_FIELD_OP(method, operator)                     \
  template <typename T>                                       \
  __global__ void method(const T* x, const T* y, T* result,   \
                         unsigned int count) {                \
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid >= count) return;                                 \
    result[gid] = x[gid] operator y[gid];                     \
  }

DEFINE_FIELD_OP(Add, +)
DEFINE_FIELD_OP(Sub, -)
DEFINE_FIELD_OP(Mul, *)
DEFINE_FIELD_OP(Div, /)

#undef DEFINE_FIELD_OP

#define DEFINE_COMPARISON_OP(method, operator)                 \
  template <typename T>                                        \
  __global__ void method(const T* x, const T* y, bool* result, \
                         unsigned int count) {                 \
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;  \
    if (gid >= count) return;                                  \
    result[gid] = x[gid] operator y[gid];                      \
  }

DEFINE_COMPARISON_OP(Eq, ==)
DEFINE_COMPARISON_OP(Ne, !=)
DEFINE_COMPARISON_OP(Lt, <)
DEFINE_COMPARISON_OP(Le, <=)
DEFINE_COMPARISON_OP(Gt, >)
DEFINE_COMPARISON_OP(Ge, >=)

#undef DEFINE_COMPARISON_OP

}  // namespace tachyon::math::kernels

#endif  // TACHYON_MATH_FINITE_FIELDS_KERNELS_PRIME_FIELD_OPS_CU_H_
