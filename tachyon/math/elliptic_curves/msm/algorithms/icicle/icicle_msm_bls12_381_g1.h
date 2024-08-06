#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BLS12_381_G1_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BLS12_381_G1_H_

#include "third_party/icicle/include/curves/params/bls12_381.cu.h"
#include "third_party/icicle/include/msm/msm.cu.h"

extern "C" cudaError_t tachyon_bls12_381_g1_msm_cuda(
    const ::bls12_381::scalar_t* scalars, const ::bls12_381::affine_t* points,
    int msm_size, ::msm::MSMConfig& config, ::bls12_381::projective_t* out);

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_BLS12_381_G1_H_
