#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BN254_G2_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BN254_G2_H_

#include "third_party/icicle/include/curves/params/bn254.cu.h"
#include "third_party/icicle/include/msm/msm.cu.h"

extern "C" cudaError_t tachyon_bn254_g2_msm_cuda(
    const ::bn254::scalar_t* scalars, const ::bn254::g2_affine_t* points,
    int msm_size, ::msm::MSMConfig& config, ::bn254::g2_projective_t* out);

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BN254_G2_H_
