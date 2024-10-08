#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BABY_BEAR_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BABY_BEAR_H_

#include <stdint.h>

#include "third_party/icicle/include/fields/stark_fields/babybear.cu.h"
#include "third_party/icicle/include/ntt/ntt.cu.h"

#include "tachyon/device/gpu/gpu_device_functions.h"

extern "C" gpuError_t tachyon_babybear_initialize_domain_cuda(
    const ::babybear::scalar_t& primitive_root,
    ::device_context::DeviceContext& ctx, bool fast_twiddles_mode);

extern "C" gpuError_t tachyon_babybear_ntt_cuda(
    const ::babybear::scalar_t* input, int size, ::ntt::NTTDir dir,
    ::ntt::NTTConfig<::babybear::scalar_t>& config,
    ::babybear::scalar_t* output);

extern "C" gpuError_t tachyon_babybear_release_domain_cuda(
    ::device_context::DeviceContext& ctx);

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BABY_BEAR_H_
