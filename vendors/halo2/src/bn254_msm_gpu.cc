#include "vendors/halo2/include/bn254_msm_gpu.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/msm_gpu.h"
#include "vendors/halo2/src/bn254.rs.h"

namespace tachyon::halo2_api::bn254 {

rust::Box<G1MSMGpu> create_g1_msm_gpu(uint8_t degree) {
  return rust::Box<G1MSMGpu>::from_raw(
      reinterpret_cast<G1MSMGpu*>(tachyon_bn254_g1_create_msm_gpu(degree)));
}

void destroy_g1_msm_gpu(rust::Box<G1MSMGpu> msm) {
  tachyon_bn254_g1_destroy_msm_gpu(
      reinterpret_cast<tachyon_bn254_g1_msm_gpu_ptr>(msm.into_raw()));
}

rust::Box<G1JacobianPoint> g1_point2_msm_gpu(G1MSMGpu* msm,
                                             rust::Slice<const G1Point2> bases,
                                             rust::Slice<const Fr> scalars) {
  auto ret = tachyon_bn254_g1_point2_msm_gpu(
      reinterpret_cast<tachyon_bn254_g1_msm_gpu_ptr>(msm),
      reinterpret_cast<const tachyon_bn254_g1_point2*>(bases.data()),
      reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
      scalars.length());
  return rust::Box<G1JacobianPoint>::from_raw(
      reinterpret_cast<G1JacobianPoint*>(ret));
}

}  // namespace tachyon::halo2_api::bn254
