#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BLS12_381_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BLS12_381_H_

#include <stdint.h>

#include "third_party/icicle/include/curves/params/bls12_381.cu.h"
#include "third_party/icicle/include/ntt/ntt.cu.h"

extern "C" cudaError_t tachyon_bls12_381_initialize_domain_cuda(
    const ::bls12_381::scalar_t& primitive_root,
    ::device_context::DeviceContext& ctx, bool fast_twiddles_mode);

extern "C" cudaError_t tachyon_bls12_381_ntt_cuda(
    const ::bls12_381::scalar_t* input, int size, ::ntt::NTTDir dir,
    ::ntt::NTTConfig<::bls12_381::scalar_t>& config,
    ::bls12_381::scalar_t* output);

extern "C" cudaError_t tachyon_bls12_381_release_domain_cuda(
    ::device_context::DeviceContext& ctx);

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_BLS12_381_H_
