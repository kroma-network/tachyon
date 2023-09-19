#include "tachyon_halo2/include/msm.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/msm.h"
#include "tachyon_halo2/src/lib.rs.h"

namespace tachyon {
namespace halo2 {

rust::Box<CppMSM> create_msm(uint8_t degree) {
  return rust::Box<CppMSM>::from_raw(
      reinterpret_cast<CppMSM*>(tachyon_bn254_g1_create_msm(degree)));
}

void destroy_msm(rust::Box<CppMSM> msm) {
  tachyon_bn254_g1_destroy_msm(
      reinterpret_cast<tachyon_bn254_g1_msm_ptr>(msm.into_raw()));
}

rust::Box<CppG1Jacobian> msm(CppMSM* msm, rust::Slice<const CppG1Affine> bases,
                             rust::Slice<const CppFr> scalars) {
  auto ret = tachyon_bn254_g1_point2_msm(
      reinterpret_cast<tachyon_bn254_g1_msm_ptr>(msm),
      reinterpret_cast<const tachyon_bn254_g1_point2*>(bases.data()),
      reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
      scalars.length());
  return rust::Box<CppG1Jacobian>::from_raw(
      reinterpret_cast<CppG1Jacobian*>(ret));
}

}  // namespace halo2
}  // namespace tachyon
