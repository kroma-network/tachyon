#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_CUDA_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_CUDA_CU_H_

#include "tachyon/math/elliptic_curves/bls/bls12_381/curve_config.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fq_cuda.cu.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fr_cuda.cu.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

#if TACHYON_CUDA
using G1AffinePointCuda = AffinePoint<CurveConfig<FqCuda, FrCuda>::Config>;
using G1JacobianPointCuda = JacobianPoint<CurveConfig<FqCuda, FrCuda>::Config>;
#endif  // TACHYON_CUDA

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_CURVE_CONFIG_CUDA_CU_H_
