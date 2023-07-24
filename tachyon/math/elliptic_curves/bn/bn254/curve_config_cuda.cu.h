#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_CUDA_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_CUDA_CU_H_

#include "tachyon/math/elliptic_curves/bn/bn254/curve_config.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq_cuda.cu.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr_cuda.cu.h"

namespace tachyon {
namespace math {
namespace bn254 {

#if TACHYON_CUDA
using G1AffinePointCuda = AffinePoint<CurveConfig<FqCuda, FrCuda>::Config>;
using G1JacobianPointCuda = JacobianPoint<CurveConfig<FqCuda, FrCuda>::Config>;
#endif  // TACHYON_CUDA

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_CONFIG_CUDA_CU_H_
