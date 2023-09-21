#ifndef TACHYON_RS_MATH_ELLIPTIC_CURVES_BN_BN254_MSM_H_
#define TACHYON_RS_MATH_ELLIPTIC_CURVES_BN_BN254_MSM_H_

#include "rust/cxx.h"

namespace tachyon::rs::math::bn254 {

struct G1MSM;

struct G1AffinePoint;
struct G1JacobianPoint;
struct G1Point2;
struct Fr;

rust::Box<G1MSM> create_g1_msm(uint8_t degree);

void destroy_g1_msm(rust::Box<G1MSM> msm);

rust::Box<G1JacobianPoint> g1_affine_msm(G1MSM* msm,
                                         rust::Slice<const G1AffinePoint> bases,
                                         rust::Slice<const Fr> scalars);

rust::Box<G1JacobianPoint> g1_point2_msm(G1MSM* msm,
                                         rust::Slice<const G1Point2> bases,
                                         rust::Slice<const Fr> scalars);

}  // namespace tachyon::rs::math::bn254

#endif  // TACHYON_RS_MATH_ELLIPTIC_CURVES_BN_BN254_MSM_H_
