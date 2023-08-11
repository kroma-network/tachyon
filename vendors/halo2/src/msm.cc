#include "tachyon_halo2/include/msm.h"

#include "tachyon/c/math/elliptic_curves/msm/msm.h"
#include "tachyon_halo2/src/lib.rs.h"

namespace tachyon {
namespace halo2 {

rust::Box<CppG1Jacobian> msm(rust::Slice<const CppG1Affine> bases,
                             rust::Slice<const CppFr> scalars) {
  auto ret = tachyon_msm_g1_point2(
      reinterpret_cast<const tachyon_bn254_point2*>(bases.data()),
      bases.length(), reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
      scalars.length());
  return rust::Box<CppG1Jacobian>::from_raw(
      reinterpret_cast<CppG1Jacobian*>(ret));
}

}  // namespace halo2
}  // namespace tachyon
