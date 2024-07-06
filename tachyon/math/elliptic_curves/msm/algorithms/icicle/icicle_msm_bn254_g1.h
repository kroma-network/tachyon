#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BN254_G1_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BN254_G1_H_

#include "third_party/icicle/include/curves/params/bn254.cu.h"
#include "third_party/icicle/include/msm/msm.cu.h"

extern "C" cudaError_t tachyon_bn254_g1_msm_cuda(
    const ::bn254::scalar_t* scalars, const ::bn254::affine_t* points,
    int msm_size, ::msm::MSMConfig& config, ::bn254::projective_t* out);

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BN254_G1_H_
