#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bn254_g1.h"

#include "third_party/icicle/src/msm/msm.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/bit_cast.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_utils.h"

gpuError_t tachyon_bn254_g1_msm_cuda(const ::bn254::scalar_t* scalars,
                                     const ::bn254::affine_t* points,
                                     int msm_size, ::msm::MSMConfig& config,
                                     ::bn254::projective_t* out) {
  return ::msm::msm(scalars, points, msm_size, config, out);
}

namespace tachyon::math {

template <>
bool IcicleMSM<bn254::G1AffinePoint>::Run(
    absl::Span<const bn254::G1AffinePoint> bases,
    absl::Span<const bn254::Fr> cpu_scalars,
    ProjectivePoint<Curve>* cpu_result) {
#if FIELD_ID != BN254
#error Only BN254 is supported
#endif

  size_t bases_size = bases.size();
  size_t scalars_size = cpu_scalars.size();

  if (bases_size != scalars_size) {
    LOG(ERROR) << "bases_size and scalars_size don't match";
    return false;
  }

  device::gpu::gpuPointerAttributes bases_attributes{};
  RETURN_AND_LOG_IF_GPU_ERROR(
      device::gpu::GpuPointerGetAttributes(&bases_attributes, bases.data()),
      "Failed to GpuPointerGetAttributes()");

  config_->are_points_on_device =
      bases_attributes.type != gpuMemoryTypeUnregistered &&
      bases_attributes.type != gpuMemoryTypeHost;

  size_t bitsize = static_cast<size_t>(
      (config_->bitsize == 0) ? ::bn254::scalar_t::NBITS : config_->bitsize);

  size_t divisions = DetermineMsmDivisionsForMemory(
      sizeof(::bn254::scalar_t), sizeof(::bn254::affine_t),
      sizeof(::bn254::projective_t), bases_size, config_->c, bitsize,
      static_cast<size_t>(config_->precompute_factor),
      static_cast<size_t>(config_->batch_size));

  size_t offset = bases_size / divisions;
  size_t remainder = bases_size % divisions;
  ::bn254::projective_t final_value = ::bn254::projective_t::zero();
  for (size_t idx = 0; idx < divisions; ++idx) {
    size_t start_idx = idx * offset;
    size_t data_size =
        ((idx == divisions - 1) && (remainder != 0)) ? remainder : offset;
    ::bn254::projective_t ret;
    gpuError_t error = tachyon_bn254_g1_msm_cuda(
        reinterpret_cast<const ::bn254::scalar_t*>(&cpu_scalars[start_idx]),
        reinterpret_cast<const ::bn254::affine_t*>(&bases[start_idx]),
        data_size, *config_, &ret);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed tachyon_bn254_g1_msm_cuda()";
      return false;
    }
    final_value = final_value + ret;
  }
  final_value = ::bn254::projective_t::to_montgomery(final_value);
  *cpu_result = base::bit_cast<ProjectivePoint<Curve>>(final_value);
  return true;
}

}  // namespace tachyon::math
