#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BN254_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BN254_H_

#include <stdint.h>

#include "third_party/icicle/include/curves/params/bn254.cu.h"
#include "third_party/icicle/include/ntt/ntt.cu.h"

#include "tachyon/device/gpu/gpu_device_functions.h"

extern "C" gpuError_t tachyon_bn254_initialize_domain_cuda(
    const ::bn254::scalar_t& primitive_root,
    ::device_context::DeviceContext& ctx, bool fast_twiddles_mode);

extern "C" gpuError_t tachyon_bn254_ntt_cuda(
    const ::bn254::scalar_t* input, int size, ::ntt::NTTDir dir,
    ::ntt::NTTConfig<::bn254::scalar_t>& config, ::bn254::scalar_t* output);

extern "C" gpuError_t tachyon_bn254_release_domain_cuda(
    ::device_context::DeviceContext& ctx);

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BN254_H_
