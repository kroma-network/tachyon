#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_BABY_BEAR_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_BABY_BEAR_H_

#include "third_party/icicle/include/fields/stark_fields/babybear.cu.h"
#include "third_party/icicle/include/poseidon2/poseidon2.cu.h"

#include "tachyon/device/gpu/gpu_device_functions.h"

extern "C" gpuError_t tachyon_babybear_poseidon2_create_cuda(
    unsigned int width, unsigned int rate, unsigned int alpha,
    unsigned int internal_rounds, unsigned int external_rounds,
    const ::babybear::scalar_t* round_constants,
    const ::babybear::scalar_t* internal_matrix_diag,
    ::poseidon2::MdsType mds_type, ::poseidon2::DiffusionStrategy diffusion,
    ::device_context::DeviceContext& ctx,
    ::poseidon2::Poseidon2<babybear::scalar_t>** poseidon);

extern "C" gpuError_t tachyon_babybear_poseidon2_load_cuda(
    unsigned int width, unsigned int rate, ::poseidon2::MdsType mds_type,
    ::poseidon2::DiffusionStrategy diffusion,
    ::device_context::DeviceContext& ctx,
    ::poseidon2::Poseidon2<babybear::scalar_t>** poseidon);

extern "C" gpuError_t tachyon_babybear_poseidon2_hash_many_cuda(
    const ::poseidon2::Poseidon2<::babybear::scalar_t>* poseidon,
    const ::babybear::scalar_t* inputs, ::babybear::scalar_t* output,
    unsigned int number_of_states, unsigned int input_block_len,
    unsigned int output_len, ::hash::HashConfig& cfg);

extern "C" gpuError_t tachyon_babybear_poseidon2_delete_cuda(
    ::poseidon2::Poseidon2<::babybear::scalar_t>* poseidon);

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_BABY_BEAR_H_
