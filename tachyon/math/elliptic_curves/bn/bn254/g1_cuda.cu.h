#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_G1_CUDA_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_G1_CUDA_CU_H_

#include "tachyon/math/elliptic_curves/bn/bn254/fq_cuda.cu.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr_cuda.cu.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_cuda.cu.h"

namespace tachyon {
namespace math {
namespace bn254 {

#if TACHYON_CUDA
using G1AffinePointCuda =
    AffinePoint<SWCurveCuda<G1CurveConfig<FqCuda, FrCuda>>>;
using G1JacobianPointCuda =
    JacobianPoint<SWCurveCuda<G1CurveConfig<FqCuda, FrCuda>>>;
using G1PointXYZZCuda = PointXYZZ<SWCurveCuda<G1CurveConfig<FqCuda, FrCuda>>>;
#endif  // TACHYON_CUDA

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_G1_CUDA_CU_H_
