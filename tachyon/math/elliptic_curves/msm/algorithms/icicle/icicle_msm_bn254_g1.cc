#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bn254_g1.h"

#include "third_party/icicle/src/msm/msm.cu.cc"  // NOLINT(build/include)

cudaError_t tachyon_bn254_g1_msm_cuda(const ::bn254::scalar_t* scalars,
                                      const ::bn254::affine_t* points,
                                      int msm_size, ::msm::MSMConfig& config,
                                      ::bn254::projective_t* out) {
  return ::msm::msm(scalars, points, msm_size, config, out);
}
