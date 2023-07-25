#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_CUDA_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_CUDA_CU_H_

#include "tachyon/math/elliptic_curves/secp/secp256k1/curve_config.h"
#include "tachyon/math/elliptic_curves/secp/secp256k1/fq_cuda.cu.h"
#include "tachyon/math/elliptic_curves/secp/secp256k1/fr_cuda.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_cuda.cu.h"

namespace tachyon {
namespace math {
namespace secp256k1 {

#if TACHYON_CUDA
using G1AffinePointCuda = AffinePoint<CurveConfigCuda<FqCuda, FrCuda>::Config>;
using G1JacobianPointCuda =
    JacobianPoint<CurveConfigCuda<FqCuda, FrCuda>::Config>;
#endif  // TACHYON_CUDA

}  // namespace secp256k1
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_CURVE_CONFIG_CUDA_CU_H_
