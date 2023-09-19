// clang-format off
#include "tachyon_halo2/include/msm.h"
// clang-format on

// clang-format off
#include "tachyon/c/math/elliptic_curves/bn/bn254/msm_gpu.h"
// clang-format on
#include "tachyon_halo2/src/lib.rs.h"

namespace tachyon {

namespace halo2 {

rust::Box<CppMSMGpu> create_msm_gpu(uint8_t degree, int algorithm) {
  return rust::Box<CppMSMGpu>::from_raw(reinterpret_cast<CppMSMGpu*>(
      tachyon_bn254_g1_create_msm_gpu(degree, algorithm)));
}

void destroy_msm_gpu(rust::Box<CppMSMGpu> msm) {
  tachyon_bn254_g1_destroy_msm_gpu(
      reinterpret_cast<tachyon_bn254_g1_msm_gpu_ptr>(msm.into_raw()));
}

rust::Box<CppG1Jacobian> msm_gpu(CppMSMGpu* msm,
                                 rust::Slice<const CppG1Affine> bases,
                                 rust::Slice<const CppFr> scalars) {
  auto ret = tachyon_bn254_g1_point2_msm_gpu(
      reinterpret_cast<tachyon_bn254_g1_msm_gpu_ptr>(msm),
      reinterpret_cast<const tachyon_bn254_g1_point2*>(bases.data()),
      reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
      scalars.length());
  return rust::Box<CppG1Jacobian>::from_raw(
      reinterpret_cast<CppG1Jacobian*>(ret));
}

}  // namespace halo2
}  // namespace tachyon
