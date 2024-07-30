#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_

#include <algorithm>
#include <memory>

#include "third_party/icicle/include/fields/id.h"

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/bits.h"
#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bn254_g1.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_bn254_g2.h"
#include "tachyon/math/geometry/projective_point.h"

namespace tachyon::math {

namespace {

// NOTE(GideokKim): The formula for memory usage estimation provided in the
// document did not match the actual memory allocation, so some of the formula
// was modified. |scalars_memory_size| and |points_memory_size| are exactly the
// same, and |scalar_indices_memory_size| internally uses the sort function of
// the cub library to set some free memory. |buckets_memory_size| uses more
// memory than the actual formula, so it was modified to an empirically more
// appropriate formula. See
// https://dev.ingonyama.com/icicle/primitives/msm#memory-usage-estimation
size_t DetermineMsmDivisionsForMemory(size_t scalar_t_mem_size,
                                      size_t affine_t_mem_size,
                                      size_t projective_t_mem_size,
                                      size_t msm_size, size_t user_c,
                                      size_t bitsize, size_t precompute_factor,
                                      size_t batch_size) {
  size_t free_memory =
      device::gpu::GpuMemLimitInfo(device::gpu::MemoryUsage::kHigh);
  size_t shift = 0;
  size_t log_msm_size = base::bits::Log2Ceiling(msm_size);

  for (size_t number_of_divisions = 0; number_of_divisions < log_msm_size;
       ++number_of_divisions) {
    // See
    // https://github.com/ingonyama-zk/icicle/blob/0cb0b49b/icicle/src/msm/msm.cu#L429-L431
    // https://github.com/ingonyama-zk/icicle/blob/0cb0b49b/icicle/include/msm/msm.cuh#L50-L56
    size_t c = (user_c == 0) ? static_cast<size_t>(std::max(
                                   base::bits::Log2Ceiling(msm_size) - 4, 1))
                             : user_c;
    size_t total_bms_per_msm = (bitsize + c - 1) / c;

    // Calculate memory requirements
    // See
    // https://github.com/ingonyama-zk/icicle/blob/0cb0b49b/icicle/src/msm/msm.cu#L408-L427
    size_t scalars_memory_size = scalar_t_mem_size * msm_size;
    // See
    // https://github.com/ingonyama-zk/icicle/blob/0cb0b49b/icicle/src/msm/msm.cu#L439-L442
    // https://github.com/ingonyama-zk/icicle/blob/0cb0b49b/icicle/src/msm/msm.cu#L461-L464
    size_t scalar_indices_memory_size = 6 * 4 * total_bms_per_msm * msm_size;
    scalar_indices_memory_size =
        static_cast<size_t>(scalar_indices_memory_size * 1.02);
    // See
    // https://github.com/ingonyama-zk/icicle/blob/0cb0b49b/icicle/src/msm/msm.cu#L515-L535
    size_t points_memory_size =
        affine_t_mem_size * precompute_factor * msm_size;
    // See
    // https://github.com/ingonyama-zk/icicle/blob/0cb0b49b/icicle/src/msm/msm.cu#L545
    // https://github.com/ingonyama-zk/icicle/blob/0cb0b49b/icicle/src/msm/msm.cu#L767-L834
    size_t buckets_memory_size =
        projective_t_mem_size * total_bms_per_msm * (size_t{3} << c);

    // Estimate total memory usage
    // See
    // https://dev.ingonyama.com/icicle/primitives/msm#memory-usage-estimation
    size_t estimated_memory =
        std::max(scalar_indices_memory_size,
                 points_memory_size + buckets_memory_size) +
        scalars_memory_size;
    estimated_memory = static_cast<size_t>(estimated_memory * batch_size * 1.1);

    if (free_memory > estimated_memory) {
      shift = number_of_divisions;
      break;
    }
    msm_size >>= 1;
  }

  return size_t{1} << shift;
}

}  // namespace

struct TACHYON_EXPORT IcicleMSMOptions {
  int points_size = 0;
  int precompute_factor = 1;
  int c = 0;
  int bitsize = 0;
  int large_bucket_factor = 10;
  int batch_size = 1;
  bool are_scalars_on_device = false;
  bool are_scalars_montgomery_form = true;
  bool are_points_on_device = false;
  bool are_points_montgomery_form = true;
  bool are_results_on_device = false;
  bool is_big_triangle = false;
  bool is_async = false;
};

template <typename Point>
class IcicleMSM {
 public:
  using Curve = typename Point::Curve;

  IcicleMSM(gpuMemPool_t mem_pool, gpuStream_t stream,
            const IcicleMSMOptions& options = IcicleMSMOptions())
      : mem_pool_(mem_pool), stream_(stream) {
    ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
    config_.reset(new ::msm::MSMConfig{
        ctx,
        options.points_size,
        options.precompute_factor,
        options.c,
        options.bitsize,
        options.large_bucket_factor,
        options.batch_size,
        options.are_scalars_on_device,
        options.are_scalars_montgomery_form,
        // TODO(chokobole): Considering KZG commitment, bases can be loaded to
        // the device just once initially.
        options.are_points_on_device,
        options.are_points_montgomery_form,
        options.are_results_on_device,
        options.is_big_triangle,
        options.is_async,
    });
    VLOG(1) << "IcicleMSM is created";
  }
  IcicleMSM(const IcicleMSM& other) = delete;
  IcicleMSM& operator=(const IcicleMSM& other) = delete;

  template <typename BaseContainer, typename ScalarContainer>
  [[nodiscard]] bool Run(const BaseContainer& cpu_bases,
                         const ScalarContainer& cpu_scalars,
                         ProjectivePoint<Curve>* cpu_result);

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  std::unique_ptr<::msm::MSMConfig> config_;
};

template <>
template <typename BaseContainer, typename ScalarContainer>
bool IcicleMSM<bn254::G1AffinePoint>::Run(const BaseContainer& cpu_bases,
                                          const ScalarContainer& cpu_scalars,
                                          ProjectivePoint<Curve>* cpu_result) {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif

  size_t bases_size = std::size(cpu_bases);
  size_t scalars_size = std::size(cpu_scalars);

  if (bases_size != scalars_size) {
    LOG(ERROR) << "bases_size and scalars_size don't match";
    return false;
  }

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
        reinterpret_cast<const ::bn254::affine_t*>(&cpu_bases[start_idx]),
        data_size, *config_, &ret);
    if (error != gpuSuccess) return false;
    final_value = final_value + ret;
  }
  final_value = ::bn254::projective_t::to_montgomery(final_value);
  *cpu_result = base::bit_cast<ProjectivePoint<Curve>>(final_value);
  return true;
}

template <>
template <typename BaseContainer, typename ScalarContainer>
bool IcicleMSM<bn254::G2AffinePoint>::Run(const BaseContainer& cpu_bases,
                                          const ScalarContainer& cpu_scalars,
                                          ProjectivePoint<Curve>* cpu_result) {
#if FIELD_ID != BN254
#error Only Bn254 is supported
#endif

  size_t bases_size = std::size(cpu_bases);
  size_t scalars_size = std::size(cpu_scalars);

  if (bases_size != scalars_size) {
    LOG(ERROR) << "bases_size and scalars_size don't match";
    return false;
  }

  size_t bitsize = static_cast<size_t>(
      (config_->bitsize == 0) ? ::bn254::scalar_t::NBITS : config_->bitsize);

  size_t divisions = DetermineMsmDivisionsForMemory(
      sizeof(::bn254::scalar_t), sizeof(::bn254::g2_affine_t),
      sizeof(::bn254::g2_projective_t), bases_size, config_->c, bitsize,
      static_cast<size_t>(config_->precompute_factor),
      static_cast<size_t>(config_->batch_size));

  size_t offset = bases_size / divisions;
  size_t remainder = bases_size % divisions;
  ::bn254::g2_projective_t final_value = ::bn254::g2_projective_t::zero();
  for (size_t idx = 0; idx < divisions; ++idx) {
    size_t start_idx = idx * offset;
    size_t data_size =
        ((idx == divisions - 1) && (remainder != 0)) ? remainder : offset;
    ::bn254::g2_projective_t ret;
    gpuError_t error = tachyon_bn254_g2_msm_cuda(
        reinterpret_cast<const ::bn254::scalar_t*>(&cpu_scalars[start_idx]),
        reinterpret_cast<const ::bn254::g2_affine_t*>(&cpu_bases[start_idx]),
        data_size, *config_, &ret);
    if (error != gpuSuccess) return false;
    final_value = final_value + ret;
  }
  final_value = ::bn254::g2_projective_t::to_montgomery(final_value);
  *cpu_result = base::bit_cast<ProjectivePoint<Curve>>(final_value);
  return true;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_H_
