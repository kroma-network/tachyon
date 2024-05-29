// clang-format off
#include <tuple>

#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_gpu.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/g1_gpu.h"

struct tachyon_%{type}_g1_msm_gpu : public tachyon::c::math::MSMGpuApi<tachyon::math::%{type}::G1CurveGpu> {
  using tachyon::c::math::MSMGpuApi<tachyon::math::%{type}::G1CurveGpu>::MSMGpuApi;
};

tachyon_%{type}_g1_msm_gpu_ptr tachyon_%{type}_g1_create_msm_gpu(uint8_t degree) {
  return new tachyon_%{type}_g1_msm_gpu(degree);
}

void tachyon_%{type}_g1_destroy_msm_gpu(tachyon_%{type}_g1_msm_gpu_ptr ptr) {
  delete ptr;
}

tachyon_%{type}_g1_jacobian* tachyon_%{type}_g1_point2_msm_gpu(
    tachyon_%{type}_g1_msm_gpu_ptr ptr, const tachyon_%{type}_g1_point2* bases,
    const tachyon_%{type}_fr* scalars, size_t size) {
  return tachyon::c::math::DoMSMGpu<tachyon::math::%{type}::G1JacobianPoint>(
      *ptr, bases, scalars, size);
}

tachyon_%{type}_g1_jacobian* tachyon_%{type}_g1_affine_msm_gpu(
    tachyon_%{type}_g1_msm_gpu_ptr ptr, const tachyon_%{type}_g1_affine* bases,
    const tachyon_%{type}_fr* scalars, size_t size) {
  return tachyon::c::math::DoMSMGpu<tachyon::math::%{type}::G1JacobianPoint>(
      *ptr, bases, scalars, size);
}
// clang-format on
