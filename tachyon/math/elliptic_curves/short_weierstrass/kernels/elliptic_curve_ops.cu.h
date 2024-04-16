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

#define DEFINE_DOUBLE_OP(src_type, dst_type)                                  \
  template <typename Config>                                                  \
  __global__ void Double(const src_type<Config>* x, dst_type<Config>* result, \
                         unsigned int count) {                                \
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (gid >= count) return;                                                 \
    result[gid] = x[gid].Double();                                            \
  }

DEFINE_DOUBLE_OP(AffinePoint, JacobianPoint)
DEFINE_DOUBLE_OP(JacobianPoint, JacobianPoint)
DEFINE_DOUBLE_OP(ProjectivePoint, ProjectivePoint)
DEFINE_DOUBLE_OP(PointXYZZ, PointXYZZ)

#undef DEFINE_DOUBLE_OP

#define DEFINE_NEGATE_OP(src_type, dst_type)                                  \
  template <typename Config>                                                  \
  __global__ void Negate(const src_type<Config>* x, dst_type<Config>* result, \
                         unsigned int count) {                                \
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;                 \
    if (gid >= count) return;                                                 \
    result[gid] = -x[gid];                                                    \
  }

DEFINE_NEGATE_OP(AffinePoint, AffinePoint)
DEFINE_NEGATE_OP(JacobianPoint, JacobianPoint)
DEFINE_NEGATE_OP(ProjectivePoint, ProjectivePoint)
DEFINE_NEGATE_OP(PointXYZZ, PointXYZZ)

#undef DEFINE_NEGATE_OP

}  // namespace tachyon::math::kernels

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_KERNELS_ELLIPTIC_CURVE_OPS_CU_H_
