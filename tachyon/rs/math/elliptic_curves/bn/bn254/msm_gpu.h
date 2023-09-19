#ifndef TACHYON_RS_MATH_ELLIPTIC_CURVES_BN_BN254_MSM_GPU_H_
#define TACHYON_RS_MATH_ELLIPTIC_CURVES_BN_BN254_MSM_GPU_H_

#include "rust/cxx.h"

namespace tachyon::rs::math::bn254 {

struct G1MSMGpu;

struct G1AffinePoint;
struct G1JacobianPoint;
struct G1Point2;
struct Fr;

rust::Box<G1MSMGpu> create_g1_msm_gpu(uint8_t degree, int algorithm);

void destroy_g1_msm_gpu(rust::Box<G1MSMGpu> msm);

rust::Box<G1JacobianPoint> g1_affine_msm_gpu(
    G1MSMGpu* msm, rust::Slice<const G1AffinePoint> bases,
    rust::Slice<const Fr> scalars);

rust::Box<G1JacobianPoint> g1_point2_msm_gpu(G1MSMGpu* msm,
                                             rust::Slice<const G1Point2> bases,
                                             rust::Slice<const Fr> scalars);

}  // namespace tachyon::rs::math::bn254

#endif  // TACHYON_RS_MATH_ELLIPTIC_CURVES_BN_BN254_MSM_GPU_H_
