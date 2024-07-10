#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt.h"

#include "third_party/icicle/include/fields/id.h"

#include "tachyon/base/bit_cast.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt_bn254.h"

namespace tachyon::math {

template <>
bool IcicleNTT<bn254::Fr>::Init(const bn254::Fr& group_gen) {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif
  ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
  math::BigInt<4> group_gen_big_int = group_gen.ToBigInt();
  gpuError_t error = tachyon_bn254_initialize_domain(
      reinterpret_cast<const ::bn254::scalar_t&>(group_gen_big_int), ctx,
      /*fast_twiddles_mode=*/true);
  if (error != gpuSuccess) return false;
  VLOG(1) << "IcicleNTT is initialized";

  config_.reset(new ::ntt::NTTConfig<bn254::Fr>{
      ctx,
      base::bit_cast<bn254::Fr>(::bn254::scalar_t::one()),
      /*batch_size=*/1,
      /*columns_batch=*/false,
      /*ordering=*/::ntt::Ordering::kNN,
      /*are_inputs_on_device=*/false,
      /*are_outputs_on_device=*/false,
      /*is_async=*/false,
      /*ntt_algorithm=*/::ntt::NttAlgorithm::Auto,
  });
  return true;
}

template <>
bool IcicleNTT<bn254::Fr>::Run(::ntt::NttAlgorithm algorithm,
                               const BigInt& coset, bn254::Fr* inout, int size,
                               ::ntt::NTTDir dir) const {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif

  // NOTE(chokobole): Manual copy is needed even though
  // |sizeof(::bn254::scalar_t)| and |sizeof(bn254::Fr)| are same. This is
  // because their alignment are different.
  // See icicle/include/fields/storage.cu.h
  ::ntt::NTTConfig<::bn254::scalar_t> config{
      config_->ctx,
      base::bit_cast<::bn254::scalar_t>(coset),
      config_->batch_size,
      config_->columns_batch,
      config_->ordering,
      config_->are_inputs_on_device,
      config_->are_outputs_on_device,
      config_->is_async,
      algorithm,
  };

  gpuError_t error = tachyon_bn254_ntt_cuda(
      reinterpret_cast<const ::bn254::scalar_t*>(inout), size, dir, config,
      reinterpret_cast<::bn254::scalar_t*>(inout));
  return error == gpuSuccess;
}

template <>
bool IcicleNTT<bn254::Fr>::Release() {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif

  ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
  gpuError_t error = tachyon_bn254_release_domain(ctx);
  return error == gpuSuccess;
}

}  // namespace tachyon::math
