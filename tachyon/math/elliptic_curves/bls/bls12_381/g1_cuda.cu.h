#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_G1_CUDA_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_G1_CUDA_CU_H_

#include "tachyon/math/elliptic_curves/bls/bls12_381/fq_cuda.cu.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fr_cuda.cu.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_cuda.cu.h"

namespace tachyon::math {
namespace bls12_381 {

#if TACHYON_CUDA
using G1AffinePointCuda =
    AffinePoint<SWCurveCuda<G1CurveConfig<FqCuda, FrCuda>>>;
using G1ProjectivePointCuda =
    ProjectivePoint<SWCurveCuda<G1CurveConfig<FqCuda, FrCuda>>>;
using G1JacobianPointCuda =
    JacobianPoint<SWCurveCuda<G1CurveConfig<FqCuda, FrCuda>>>;
using G1PointXYZZCuda = PointXYZZ<SWCurveCuda<G1CurveConfig<FqCuda, FrCuda>>>;
#endif  // TACHYON_CUDA

}  // namespace bls12_381
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_G1_CUDA_CU_H_
