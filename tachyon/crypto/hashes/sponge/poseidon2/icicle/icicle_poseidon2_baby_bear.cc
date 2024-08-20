#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2_baby_bear.h"

#include "third_party/icicle/include/fields/id.h"
#include "third_party/icicle/include/gpu-utils/error_handler.cu.h"
#include "third_party/icicle/src/poseidon2/constants.cu.cc"  // NOLINT(build/include)

#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"

cudaError_t tachyon_babybear_poseidon2_create_cuda(
    ::poseidon2::Poseidon2<::babybear::scalar_t>** poseidon, unsigned int width,
    unsigned int rate, unsigned int alpha, unsigned int internal_rounds,
    unsigned int external_rounds, const ::babybear::scalar_t* round_constants,
    const ::babybear::scalar_t* internal_matrix_diag,
    ::poseidon2::MdsType mds_type, ::poseidon2::DiffusionStrategy diffusion,
    ::device_context::DeviceContext& ctx) {
  try {
    *poseidon = new ::poseidon2::Poseidon2<::babybear::scalar_t>(
        width, rate, alpha, internal_rounds, external_rounds, round_constants,
        internal_matrix_diag, mds_type, diffusion, ctx);
    return cudaError_t::cudaSuccess;
  } catch (const ::IcicleError& _error) {
    return cudaError_t::cudaErrorUnknown;
  }
}

cudaError_t tachyon_babybear_poseidon2_load_cuda(
    ::poseidon2::Poseidon2<::babybear::scalar_t>** poseidon, unsigned int width,
    unsigned int rate, ::poseidon2::MdsType mds_type,
    ::poseidon2::DiffusionStrategy diffusion,
    ::device_context::DeviceContext& ctx) {
  try {
    *poseidon = new ::poseidon2::Poseidon2<::babybear::scalar_t>(
        width, rate, mds_type, diffusion, ctx);
    return cudaError_t::cudaSuccess;
  } catch (const IcicleError& _error) {
    return cudaError_t::cudaErrorUnknown;
  }
}

cudaError_t tachyon_babybear_poseidon2_hash_many_cuda(
    const ::poseidon2::Poseidon2<::babybear::scalar_t>* poseidon,
    const ::babybear::scalar_t* inputs, ::babybear::scalar_t* output,
    unsigned int number_of_states, unsigned int input_block_len,
    unsigned int output_len, ::hash::HashConfig& cfg) {
  return poseidon->hash_many(inputs, output, number_of_states, input_block_len,
                             output_len, cfg);
}

cudaError_t tachyon_babybear_poseidon2_delete_cuda(
    ::poseidon2::Poseidon2<::babybear::scalar_t>* poseidon) {
  try {
    poseidon->~Poseidon2();
    return cudaError_t::cudaSuccess;
  } catch (const IcicleError& _error) {
    return cudaError_t::cudaErrorUnknown;
  }
}

namespace tachyon::crypto {

namespace {

bool configure_mds_and_diffusion_strategy(
    Vendor type, ::poseidon2::MdsType& mds_type,
    ::poseidon2::DiffusionStrategy& diffusion_strategy) {
  switch (type) {
    case Vendor::kHorizen:
      mds_type = ::poseidon2::MdsType::DEFAULT_MDS;
      diffusion_strategy = ::poseidon2::DiffusionStrategy::DEFAULT_DIFFUSION;
      break;
    case Vendor::kPlonky3:
      mds_type = ::poseidon2::MdsType::PLONKY;
      diffusion_strategy = ::poseidon2::DiffusionStrategy::MONTGOMERY;
      break;
    default:
      NOTREACHED();
      return false;
  }
  return true;
}

}  // namespace

template <>
bool IciclePoseidon2<math::BabyBear>::Create(
    unsigned int width, unsigned int rate, unsigned int alpha,
    unsigned int internal_rounds, unsigned int external_rounds,
    absl::Span<const math::BabyBear> round_constants,
    absl::Span<const math::BabyBear> internal_matrix_diag, Vendor type) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  if (poseidon_ != nullptr) {
    VLOG(1) << "IciclePoseidon2 was already initialized";
    return true;
  }

  ::poseidon2::MdsType mds_type;
  ::poseidon2::DiffusionStrategy diffusion_strategy;

  if (!configure_mds_and_diffusion_strategy(type, mds_type,
                                            diffusion_strategy)) {
    return false;
  }

  auto convert_to_scalar_t = [](const absl::Span<const math::BabyBear>& span) {
    auto ptr = std::make_unique<::babybear::scalar_t[]>(span.size());
    for (size_t i = 0; i < span.size(); ++i) {
      ptr[i] = ::babybear::scalar_t::from_montgomery(
          reinterpret_cast<const ::babybear::scalar_t*>(std::data(span))[i]);
    }
    return ptr;
  };

  auto round_constants_ptr = convert_to_scalar_t(round_constants);
  auto internal_matrix_diag_ptr = convert_to_scalar_t(internal_matrix_diag);

  ::poseidon2::Poseidon2<::babybear::scalar_t>* icicle_poseidon = nullptr;

  gpuError_t error = tachyon_babybear_poseidon2_create_cuda(
      &icicle_poseidon, width, rate, alpha, internal_rounds, external_rounds,
      round_constants_ptr.get(), internal_matrix_diag_ptr.get(), mds_type,
      diffusion_strategy, config_->ctx);

  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_babybear_poseidon2_create_cuda()";
    return false;
  }

  poseidon_ = static_cast<void*>(icicle_poseidon);
  return true;
}

template <>
bool IciclePoseidon2<math::BabyBear>::Load(unsigned int width,
                                           unsigned int rate, Vendor type) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  if (poseidon_ != nullptr) {
    VLOG(1) << "IciclePoseidon2 was already initialized";
    return true;
  }

  ::poseidon2::MdsType mds_type;
  ::poseidon2::DiffusionStrategy diffusion_strategy;

  if (!configure_mds_and_diffusion_strategy(type, mds_type,
                                            diffusion_strategy)) {
    return false;
  }

  ::poseidon2::Poseidon2<::babybear::scalar_t>* icicle_poseidon = nullptr;

  gpuError_t error = tachyon_babybear_poseidon2_load_cuda(
      &icicle_poseidon, width, rate, mds_type, diffusion_strategy,
      config_->ctx);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_babybear_poseidon2_load_cuda()";
    return false;
  }

  poseidon_ = static_cast<void*>(icicle_poseidon);
  return true;
}

template <>
bool IciclePoseidon2<math::BabyBear>::Hash(
    absl::Span<const math::BabyBear> inputs, math::BabyBear* output,
    unsigned int number_of_states, unsigned int input_block_len,
    unsigned int output_len) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  if (poseidon_ == nullptr) {
    VLOG(1) << "IciclePoseidon2 is not initialized";
    return false;
  }

  auto allocate_and_convert_input =
      [](const absl::Span<const math::BabyBear>& inputs, size_t total_size) {
        std::unique_ptr<::babybear::scalar_t[]> ptr(
            new ::babybear::scalar_t[total_size]);
        for (size_t i = 0; i < total_size; ++i) {
          ptr[i] = ::babybear::scalar_t::from_montgomery(
              reinterpret_cast<const ::babybear::scalar_t*>(
                  std::data(inputs))[i]);
        }
        return ptr;
      };

  size_t input_size = number_of_states * input_block_len;
  size_t output_size = number_of_states * output_len;

  auto in_ptr = allocate_and_convert_input(inputs, input_size);
  std::unique_ptr<::babybear::scalar_t[]> ret(
      new ::babybear::scalar_t[output_size]);

  gpuError_t error = tachyon_babybear_poseidon2_hash_many_cuda(
      reinterpret_cast<::poseidon2::Poseidon2<::babybear::scalar_t>*>(
          poseidon_),
      in_ptr.get(), ret.get(), number_of_states, input_block_len, output_len,
      *config_);

  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error)
        << "Failed tachyon_babybear_poseidon2_hash_many_cuda()";
    return false;
  }

  for (size_t idx = 0; idx < output_size; ++idx) {
    ret[idx] = ::babybear::scalar_t::to_montgomery(ret[idx]);
    output[idx] = *reinterpret_cast<math::BabyBear*>(&ret[idx]);
  }

  return true;
}

template <>
bool IciclePoseidon2<math::BabyBear>::Delete() {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  if (!poseidon_) {
    VLOG(1) << "IciclePoseidon2 is not initialized";
    return false;
  }
  gpuError_t error = tachyon_babybear_poseidon2_delete_cuda(
      reinterpret_cast<::poseidon2::Poseidon2<::babybear::scalar_t>*>(
          poseidon_));
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_babybear_poseidon2_delete_cuda()";
    return false;
  }
  poseidon_ = nullptr;
  return true;
}

}  // namespace tachyon::crypto
