#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt_bn254.h"

#include "third_party/icicle/src/ntt/ntt.cu.cc"  // NOLINT(build/include)

cudaError_t tachyon_bn254_initialize_domain(
    const ::bn254::scalar_t& primitive_root,
    ::device_context::DeviceContext& ctx, bool fast_twiddles_mode) {
  return ::ntt::init_domain(primitive_root, ctx, fast_twiddles_mode);
}

cudaError_t tachyon_bn254_ntt_cuda(const ::bn254::scalar_t* input, int size,
                                   ::ntt::NTTDir dir,
                                   ::ntt::NTTConfig<::bn254::scalar_t>& config,
                                   ::bn254::scalar_t* output) {
  return ::ntt::ntt(input, size, dir, config, output);
}

cudaError_t tachyon_bn254_release_domain(::device_context::DeviceContext& ctx) {
  return ::ntt::release_domain<::bn254::scalar_t>(ctx);
}
