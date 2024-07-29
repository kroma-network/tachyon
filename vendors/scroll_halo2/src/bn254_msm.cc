#include "vendors/scroll_halo2/include/bn254_msm.h"

#include <memory>

#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/msm.h"
#include "vendors/scroll_halo2/src/bn254.rs.h"

namespace tachyon::halo2_api::bn254 {

rust::Box<G1MSM> create_g1_msm(uint8_t degree) {
  return rust::Box<G1MSM>::from_raw(
      reinterpret_cast<G1MSM*>(tachyon_bn254_g1_create_msm(degree)));
}

void destroy_g1_msm(rust::Box<G1MSM> msm) {
  tachyon_bn254_g1_destroy_msm(
      reinterpret_cast<tachyon_bn254_g1_msm_ptr>(msm.into_raw()));
}

rust::Box<G1ProjectivePoint> g1_point2_msm(G1MSM* msm,
                                           rust::Slice<const G1Point2> bases,
                                           rust::Slice<const Fr> scalars) {
  std::unique_ptr<tachyon_bn254_g1_jacobian> jacobian(
      tachyon_bn254_g1_point2_msm(
          reinterpret_cast<tachyon_bn254_g1_msm_ptr>(msm),
          reinterpret_cast<const tachyon_bn254_g1_point2*>(bases.data()),
          reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
          scalars.length()));
  auto ret = new math::bn254::G1ProjectivePoint(
      c::base::native_cast(jacobian.get())->ToProjective());
  return rust::Box<G1ProjectivePoint>::from_raw(
      reinterpret_cast<G1ProjectivePoint*>(ret));
}

}  // namespace tachyon::halo2_api::bn254
