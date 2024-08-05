#ifndef VENDORS_SCROLL_HALO2_INCLUDE_BN254_MSM_GPU_H_
#define VENDORS_SCROLL_HALO2_INCLUDE_BN254_MSM_GPU_H_

#include "rust/cxx.h"

namespace tachyon::halo2_api::bn254 {

struct G1MSMGpu;

struct G1ProjectivePoint;
struct G1Point2;
struct Fr;

rust::Box<G1MSMGpu> create_g1_msm_gpu(uint8_t degree);

void destroy_g1_msm_gpu(rust::Box<G1MSMGpu> msm);

rust::Box<G1ProjectivePoint> g1_point2_msm_gpu(
    G1MSMGpu* msm, rust::Slice<const G1Point2> bases,
    rust::Slice<const Fr> scalars);

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_SCROLL_HALO2_INCLUDE_BN254_MSM_GPU_H_
