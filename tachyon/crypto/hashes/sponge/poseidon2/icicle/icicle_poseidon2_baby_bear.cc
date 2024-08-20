#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2_baby_bear.h"

#include <memory>
#include <vector>

#include "third_party/icicle/include/fields/id.h"
#include "third_party/icicle/include/gpu-utils/error_handler.cu.h"
#include "third_party/icicle/src/poseidon2/constants.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"

gpuError_t tachyon_babybear_poseidon2_create_cuda(
    unsigned int width, unsigned int rate, unsigned int alpha,
    unsigned int internal_rounds, unsigned int external_rounds,
    const ::babybear::scalar_t* round_constants,
    const ::babybear::scalar_t* internal_matrix_diag,
    ::poseidon2::MdsType mds_type, ::poseidon2::DiffusionStrategy diffusion,
    ::device_context::DeviceContext& ctx,
    ::poseidon2::Poseidon2<::babybear::scalar_t>** poseidon) {
  try {
    *poseidon = new ::poseidon2::Poseidon2<::babybear::scalar_t>(
        width, rate, alpha, internal_rounds, external_rounds, round_constants,
        internal_matrix_diag, mds_type, diffusion, ctx);
    return gpuSuccess;
  } catch (const ::IcicleError& error) {
    LOG(ERROR) << "Failed tachyon_babybear_poseidon2_create_cuda(): "
               << error.what();
    return cudaErrorUnknown;
  }
}

gpuError_t tachyon_babybear_poseidon2_load_cuda(
    unsigned int width, unsigned int rate, ::poseidon2::MdsType mds_type,
    ::poseidon2::DiffusionStrategy diffusion,
    ::device_context::DeviceContext& ctx,
    ::poseidon2::Poseidon2<::babybear::scalar_t>** poseidon) {
  try {
    *poseidon = new ::poseidon2::Poseidon2<::babybear::scalar_t>(
        width, rate, mds_type, diffusion, ctx);
    return gpuSuccess;
  } catch (const IcicleError& error) {
    LOG(ERROR) << "Failed tachyon_babybear_poseidon2_load_cuda(): "
               << error.what();
    return cudaErrorUnknown;
  }
}

gpuError_t tachyon_babybear_poseidon2_hash_many_cuda(
    const ::poseidon2::Poseidon2<::babybear::scalar_t>* poseidon,
    const ::babybear::scalar_t* inputs, ::babybear::scalar_t* output,
    unsigned int number_of_states, unsigned int input_block_len,
    unsigned int output_len, ::hash::HashConfig& cfg) {
  return poseidon->hash_many(inputs, output, number_of_states, input_block_len,
                             output_len, cfg);
}

gpuError_t tachyon_babybear_poseidon2_delete_cuda(
    ::poseidon2::Poseidon2<::babybear::scalar_t>* poseidon) {
  try {
    delete poseidon;
    return gpuSuccess;
  } catch (const IcicleError& error) {
    LOG(ERROR) << "Failed tachyon_babybear_poseidon2_delete_cuda(): "
               << error.what();
    return cudaErrorUnknown;
  }
}

namespace tachyon::crypto {

template <>
bool IciclePoseidon2<math::BabyBear>::Create(
    unsigned int rate, unsigned int width, unsigned int alpha,
    unsigned int external_rounds, unsigned int internal_rounds,
    Poseidon2Vendor external_matrix_vendor,
    Poseidon2Vendor internal_matrix_vendor,
    absl::Span<const math::BabyBear> round_constants,
    absl::Span<const math::BabyBear> internal_matrix_diag) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  if (impl_ != nullptr) {
    VLOG(1) << "IciclePoseidon2 was already initialized";
    return true;
  }

  std::vector<::babybear::scalar_t> round_constants_tmp =
      base::Map(round_constants, [](const math::BabyBear& f) {
        return ::babybear::scalar_t::from_montgomery(
            reinterpret_cast<const ::babybear::scalar_t&>(f));
      });
  std::vector<::babybear::scalar_t> internal_matrix_diag_tmp =
      base::Map(internal_matrix_diag, [](const math::BabyBear& f) {
        return ::babybear::scalar_t::from_montgomery(
            reinterpret_cast<const ::babybear::scalar_t&>(f));
      });
  ::poseidon2::Poseidon2<::babybear::scalar_t>* icicle_poseidon = nullptr;
  gpuError_t error = tachyon_babybear_poseidon2_create_cuda(
      width, rate, alpha, internal_rounds, external_rounds,
      round_constants_tmp.data(), internal_matrix_diag_tmp.data(),
      external_matrix_vendor == Poseidon2Vendor::kHorizen
          ? ::poseidon2::DEFAULT_MDS
          : ::poseidon2::PLONKY,
      internal_matrix_vendor == Poseidon2Vendor::kHorizen
          ? ::poseidon2::DEFAULT_DIFFUSION
          : ::poseidon2::MONTGOMERY,
      config_->ctx, &icicle_poseidon);
  if (error != gpuSuccess) return false;

  impl_ = static_cast<void*>(icicle_poseidon);
  return true;
}

template <>
bool IciclePoseidon2<math::BabyBear>::Load(
    unsigned int rate, unsigned int width,
    Poseidon2Vendor external_matrix_vendor,
    Poseidon2Vendor internal_matrix_vendor) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  if (impl_ != nullptr) {
    VLOG(1) << "IciclePoseidon2 was already initialized";
    return true;
  }

  ::poseidon2::Poseidon2<::babybear::scalar_t>* icicle_poseidon = nullptr;
  gpuError_t error = tachyon_babybear_poseidon2_load_cuda(
      width, rate,
      external_matrix_vendor == Poseidon2Vendor::kHorizen
          ? ::poseidon2::DEFAULT_MDS
          : ::poseidon2::PLONKY,
      internal_matrix_vendor == Poseidon2Vendor::kHorizen
          ? ::poseidon2::DEFAULT_DIFFUSION
          : ::poseidon2::MONTGOMERY,
      config_->ctx, &icicle_poseidon);
  if (error != gpuSuccess) return false;

  impl_ = static_cast<void*>(icicle_poseidon);
  return true;
}

template <>
bool IciclePoseidon2<math::BabyBear>::Hash(
    unsigned int rate, absl::Span<const math::BabyBear> inputs,
    absl::Span<math::BabyBear> outputs) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  if (impl_ == nullptr) {
    VLOG(1) << "IciclePoseidon2 is not initialized";
    return false;
  }

  // TODO(chokobole): Change it to allocate just once across many |Hash()|
  // calls.
  std::vector<::babybear::scalar_t> inputs_tmp(inputs.size());
  OMP_PARALLEL_FOR(size_t i = 0; i < inputs_tmp.size(); ++i) {
    inputs_tmp[i] = ::babybear::scalar_t::from_montgomery(
        reinterpret_cast<const ::babybear::scalar_t&>(inputs[i]));
  }

  size_t num_states = inputs.size() / rate;
  gpuError_t error = tachyon_babybear_poseidon2_hash_many_cuda(
      reinterpret_cast<::poseidon2::Poseidon2<::babybear::scalar_t>*>(impl_),
      inputs_tmp.data(),
      reinterpret_cast<::babybear::scalar_t*>(outputs.data()), num_states, rate,
      outputs.size() / num_states, *config_);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error)
        << "Failed tachyon_babybear_poseidon2_hash_many_cuda()";
    return false;
  }

  OMP_PARALLEL_FOR(size_t i = 0; i < outputs.size(); ++i) {
    reinterpret_cast<::babybear::scalar_t&>(outputs[i]) =
        ::babybear::scalar_t::to_montgomery(
            reinterpret_cast<const ::babybear::scalar_t&>(outputs[i]));
  }
  return true;
}

template <>
bool IciclePoseidon2<math::BabyBear>::Delete() {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  if (!impl_) {
    VLOG(1) << "IciclePoseidon2 is not initialized";
    return false;
  }
  gpuError_t error = tachyon_babybear_poseidon2_delete_cuda(
      reinterpret_cast<::poseidon2::Poseidon2<::babybear::scalar_t>*>(
          std::exchange(impl_, nullptr)));
  return error == gpuSuccess;
}

}  // namespace tachyon::crypto
