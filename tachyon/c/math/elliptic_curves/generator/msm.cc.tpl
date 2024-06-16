// clang-format off
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/msm/msm.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/g1.h"

struct tachyon_%{type}_g1_msm : public tachyon::c::math::MSMApi<tachyon::math::%{type}::G1AffinePoint> {
  using tachyon::c::math::MSMApi<tachyon::math::%{type}::G1AffinePoint>::MSMApi;
};

tachyon_%{type}_g1_msm_ptr tachyon_%{type}_g1_create_msm(uint8_t degree) {
  return new tachyon_%{type}_g1_msm(degree);
}

void tachyon_%{type}_g1_destroy_msm(tachyon_%{type}_g1_msm_ptr ptr) {
  delete ptr;
}

tachyon_%{type}_g1_jacobian* tachyon_%{type}_g1_point2_msm(
    tachyon_%{type}_g1_msm_ptr ptr, const tachyon_%{type}_g1_point2* bases,
    const tachyon_%{type}_fr* scalars, size_t size) {
  return tachyon::c::math::DoMSM<tachyon::math::%{type}::G1JacobianPoint>(
      *ptr, bases, scalars, size);
}

tachyon_%{type}_g1_jacobian* tachyon_%{type}_g1_affine_msm(
    tachyon_%{type}_g1_msm_ptr ptr, const tachyon_%{type}_g1_affine* bases,
    const tachyon_%{type}_fr* scalars, size_t size) {
  return tachyon::c::math::DoMSM<tachyon::math::%{type}::G1JacobianPoint>(
      *ptr, bases, scalars, size);
}
// clang-format on
