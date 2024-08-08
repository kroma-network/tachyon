#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt_bls12_381.h"

#include "third_party/icicle/include/fields/id.h"
#include "third_party/icicle/src/ntt/ntt.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/bit_cast.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt.h"

gpuError_t tachyon_bls12_381_initialize_domain_cuda(
    const ::bls12_381::scalar_t& primitive_root,
    ::device_context::DeviceContext& ctx, bool fast_twiddles_mode) {
  return ::ntt::init_domain(primitive_root, ctx, fast_twiddles_mode);
}

gpuError_t tachyon_bls12_381_ntt_cuda(
    const ::bls12_381::scalar_t* input, int size, ::ntt::NTTDir dir,
    ::ntt::NTTConfig<::bls12_381::scalar_t>& config,
    ::bls12_381::scalar_t* output) {
  return ::ntt::ntt(input, size, dir, config, output);
}

gpuError_t tachyon_bls12_381_release_domain_cuda(
    ::device_context::DeviceContext& ctx) {
  return ::ntt::release_domain<::bls12_381::scalar_t>(ctx);
}

namespace tachyon::math {

template <>
bool IcicleNTT<bls12_381::Fr>::Init(const bls12_381::Fr& group_gen,
                                    const IcicleNTTOptions& options) {
#if FIELD_ID != BLS12_381
#error Only BLS12_381 is supported
#endif
  ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
  math::BigInt<4> group_gen_big_int = group_gen.ToBigInt();
  // TODO(chokobole): We must handle these issues with domain initialization:
  // 1. It gets too slow when the domain size is 1, 2, 4, or small in general.
  //    See "vendors/circom/prover_main.cc".
  // 2. |fast_twiddles_mode| consumes a lot of memory, so we need to disable it
  //    if the ram of the GPU is not enough. See
  //    https://github.com/ingonyama-zk/icicle/blob/4fef542/icicle/include/ntt/ntt.cuh#L26-L40.
  gpuError_t error = tachyon_bls12_381_initialize_domain_cuda(
      reinterpret_cast<const ::bls12_381::scalar_t&>(group_gen_big_int), ctx,
      options.fast_twiddles_mode);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error)
        << "Failed tachyon_bls12_381_initialize_domain_cuda()";
    return false;
  }
  VLOG(1) << "IcicleNTT is initialized";

  auto one = ::bls12_381::scalar_t::one();
  config_.reset(new ::ntt::NTTConfig<bls12_381::Fr>{
      ctx,
      // TODO(chokobole): Change it to |base::bit_cast| again if the
      // |::bls12_381::scalar_t| becomes trivially copyable.
      *reinterpret_cast<bls12_381::Fr*>(&one),
      options.batch_size,
      options.columns_batch,
      options.ordering,
      options.are_inputs_on_device,
      options.are_outputs_on_device,
      options.is_async,
      /*ntt_algorithm=*/::ntt::NttAlgorithm::Auto,
  });
  return true;
}

template <>
bool IcicleNTT<bls12_381::Fr>::Run(::ntt::NttAlgorithm algorithm,
                                   const BigInt& coset, bls12_381::Fr* inout,
                                   int size, ::ntt::NTTDir dir) const {
#if FIELD_ID != BLS12_381
#error Only BLS12_381 is supported
#endif

  // NOTE(chokobole): Manual copy is needed even though
  // |sizeof(::bls12_381::scalar_t)| and |sizeof(bls12_381::Fr)| are same. This
  // is because their alignments are different. See
  // https://github.com/ingonyama-zk/icicle/blob/4fef542/icicle/include/fields/storage.cuh.
  ::ntt::NTTConfig<::bls12_381::scalar_t> config{
      config_->ctx,
      // TODO(chokobole): Change it to |base::bit_cast| again if the
      // |::bls12_381::scalar_t| becomes trivially copyable.
      *reinterpret_cast<::bls12_381::scalar_t*>(const_cast<BigInt*>(&coset)),
      config_->batch_size,
      config_->columns_batch,
      config_->ordering,
      config_->are_inputs_on_device,
      config_->are_outputs_on_device,
      config_->is_async,
      algorithm,
  };

  gpuError_t error = tachyon_bls12_381_ntt_cuda(
      reinterpret_cast<const ::bls12_381::scalar_t*>(inout), size, dir, config,
      reinterpret_cast<::bls12_381::scalar_t*>(inout));
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_bls12_381_ntt_cuda()";
    return false;
  }
  return true;
}

template <>
bool IcicleNTT<bls12_381::Fr>::Release() {
#if FIELD_ID != BLS12_381
#error Only BLS12_381 is supported
#endif

  ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
  gpuError_t error = tachyon_bls12_381_release_domain_cuda(ctx);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_bls12_381_release_domain_cuda()";
    return false;
  }
  return true;
}

}  // namespace tachyon::math
