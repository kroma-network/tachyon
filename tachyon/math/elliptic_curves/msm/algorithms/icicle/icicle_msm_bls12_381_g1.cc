#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bls12_381_g1.h"

#include "third_party/icicle/src/msm/msm.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/bit_cast.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_utils.h"

cudaError_t tachyon_bls12_381_g1_msm_cuda(const ::bls12_381::scalar_t* scalars,
                                          const ::bls12_381::affine_t* points,
                                          int msm_size,
                                          ::msm::MSMConfig& config,
                                          ::bls12_381::projective_t* out) {
  return ::msm::msm(scalars, points, msm_size, config, out);
}

namespace tachyon::math {

template <>
bool IcicleMSM<bls12_381::G1AffinePoint>::Run(
    absl::Span<const bls12_381::G1AffinePoint> cpu_bases,
    absl::Span<const bls12_381::Fr> cpu_scalars,
    ProjectivePoint<Curve>* cpu_result) {
#if FIELD_ID != BLS12_381
#error Only BLS12_381 is supported
#endif

  size_t bases_size = cpu_bases.size();
  size_t scalars_size = cpu_scalars.size();

  if (bases_size != scalars_size) {
    LOG(ERROR) << "bases_size and scalars_size don't match";
    return false;
  }

  size_t bitsize =
      static_cast<size_t>((config_->bitsize == 0) ? ::bls12_381::scalar_t::NBITS
                                                  : config_->bitsize);

  size_t divisions = DetermineMsmDivisionsForMemory(
      sizeof(::bls12_381::scalar_t), sizeof(::bls12_381::affine_t),
      sizeof(::bls12_381::projective_t), bases_size, config_->c, bitsize,
      static_cast<size_t>(config_->precompute_factor),
      static_cast<size_t>(config_->batch_size));

  size_t offset = bases_size / divisions;
  size_t remainder = bases_size % divisions;
  ::bls12_381::projective_t final_value = ::bls12_381::projective_t::zero();
  for (size_t idx = 0; idx < divisions; ++idx) {
    size_t start_idx = idx * offset;
    size_t data_size =
        ((idx == divisions - 1) && (remainder != 0)) ? remainder : offset;
    ::bls12_381::projective_t ret;
    gpuError_t error = tachyon_bls12_381_g1_msm_cuda(
        reinterpret_cast<const ::bls12_381::scalar_t*>(&cpu_scalars[start_idx]),
        reinterpret_cast<const ::bls12_381::affine_t*>(&cpu_bases[start_idx]),
        data_size, *config_, &ret);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to tachyon_bls12_381_g1_msm_cuda()";
      return false;
    }
    final_value = final_value + ret;
  }
  final_value = ::bls12_381::projective_t::to_montgomery(final_value);
  *cpu_result = base::bit_cast<ProjectivePoint<Curve>>(final_value);
  return true;
}

}  // namespace tachyon::math
