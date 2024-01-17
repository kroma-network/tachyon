#ifndef VENDORS_HALO2_INCLUDE_BN254_MSM_H_
#define VENDORS_HALO2_INCLUDE_BN254_MSM_H_

#include "rust/cxx.h"

namespace tachyon::halo2_api::bn254 {

struct G1MSM;

struct G1JacobianPoint;
struct G1Point2;
struct Fr;

rust::Box<G1MSM> create_g1_msm(uint8_t degree);

void destroy_g1_msm(rust::Box<G1MSM> msm);

rust::Box<G1JacobianPoint> g1_point2_msm(G1MSM* msm,
                                         rust::Slice<const G1Point2> bases,
                                         rust::Slice<const Fr> scalars);

void create_proof(uint8_t degree);

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_MSM_H_
