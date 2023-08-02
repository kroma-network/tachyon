#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_KERNELS_ELLIPTIC_CURVE_OPS_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_KERNELS_ELLIPTIC_CURVE_OPS_CU_H_

#include <stddef.h>

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"

namespace tachyon::math::kernels {

#define DEFINE_FIELD_OP(method, operator, src_type, dst_type)                  \
  template <typename Config>                                                   \
  __global__ void method(const src_type<Config>* x, const src_type<Config>* y, \
                         dst_type<Config>* result, unsigned int count) {       \
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;                  \
    if (gid >= count) return;                                                  \
    result[gid] = x[gid] operator y[gid];                                      \
  }

DEFINE_FIELD_OP(Add, +, AffinePoint, JacobianPoint)
DEFINE_FIELD_OP(Add, +, ProjectivePoint, ProjectivePoint)
DEFINE_FIELD_OP(Add, +, JacobianPoint, JacobianPoint)
DEFINE_FIELD_OP(Add, +, PointXYZZ, PointXYZZ)

#undef DEFINE_FIELD_OP

#define DEFINE_COMPARISON_OP(method, operator, type)                   \
  template <typename Config>                                           \
  __global__ void method(const type<Config>* x, const type<Config>* y, \
                         bool* result, unsigned int count) {           \
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;          \
    if (gid >= count) return;                                          \
    result[gid] = x[gid] operator y[gid];                              \
  }

DEFINE_COMPARISON_OP(Eq, ==, AffinePoint)
DEFINE_COMPARISON_OP(Ne, !=, AffinePoint)
DEFINE_COMPARISON_OP(Eq, ==, ProjectivePoint)
DEFINE_COMPARISON_OP(Ne, !=, ProjectivePoint)
DEFINE_COMPARISON_OP(Eq, ==, JacobianPoint)
DEFINE_COMPARISON_OP(Ne, !=, JacobianPoint)
DEFINE_COMPARISON_OP(Eq, ==, PointXYZZ)
DEFINE_COMPARISON_OP(Ne, !=, PointXYZZ)

#undef DEFINE_COMPARISON_OP

#define DEFINE_UNARY_OP(method, src_type, dst_type)                           \
  template <typename Config>                                                  \
  __global__ void method(const src_type<Config>* x, dst_type<Config>* result, \
                         unsigned int count) {                                \
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (gid >= count) return;                                                 \
    result[gid] = x[gid].method();                                            \
  }

DEFINE_UNARY_OP(Double, AffinePoint, JacobianPoint)
DEFINE_UNARY_OP(Double, JacobianPoint, JacobianPoint)
DEFINE_UNARY_OP(Double, ProjectivePoint, ProjectivePoint)
DEFINE_UNARY_OP(Double, PointXYZZ, PointXYZZ)
DEFINE_UNARY_OP(Negative, AffinePoint, AffinePoint)
DEFINE_UNARY_OP(Negative, JacobianPoint, JacobianPoint)
DEFINE_UNARY_OP(Negative, ProjectivePoint, ProjectivePoint)
DEFINE_UNARY_OP(Negative, PointXYZZ, PointXYZZ)

#undef DEFINE_UNARY_OP

}  // namespace tachyon::math::kernels

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_KERNELS_ELLIPTIC_CURVE_OPS_CU_H_
