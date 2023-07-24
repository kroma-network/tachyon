#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_KERNELS_ELLIPTIC_CURVE_OPS_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_KERNELS_ELLIPTIC_CURVE_OPS_CU_H_

#include <stddef.h>

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

namespace tachyon {
namespace math {
namespace kernels {

#define DEFINE_FIELD_OP(method, operator)                               \
  template <typename Config>                                            \
  __global__ void method(const JacobianPoint<Config>* x,                \
                         const JacobianPoint<Config>* y,                \
                         JacobianPoint<Config>* result, size_t count) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (gid >= count) return;                                           \
    result[gid] = x[gid] operator y[gid];                               \
  }

DEFINE_FIELD_OP(Add, +)

#undef DEFINE_FIELD_OP

#define DEFINE_COMPARISON_OP(method, operator)                         \
  template <typename Config>                                           \
  __global__ void method(const JacobianPoint<Config>* x,               \
                         const JacobianPoint<Config>* y, bool* result, \
                         size_t count) {                               \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (gid >= count) return;                                          \
    result[gid] = x[gid] operator y[gid];                              \
  }

DEFINE_COMPARISON_OP(Eq, ==)
DEFINE_COMPARISON_OP(Ne, !=)

#undef DEFINE_COMPARISON_OP

#define DEFINE_UNARY_OP(method)                                         \
  template <typename Config>                                            \
  __global__ void method(const JacobianPoint<Config>* x,                \
                         JacobianPoint<Config>* result, size_t count) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (gid >= count) return;                                           \
    result[gid] = x[gid].method();                                      \
  }

DEFINE_UNARY_OP(Double)
DEFINE_UNARY_OP(Negative)

#undef DEFINE_UNARY_OP

}  // namespace kernels
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_KERNELS_ELLIPTIC_CURVE_OPS_CU_H_
