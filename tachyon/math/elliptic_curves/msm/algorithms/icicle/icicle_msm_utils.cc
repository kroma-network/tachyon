#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm_utils.h"

#include <algorithm>

#include "tachyon/base/bits.h"
#include "tachyon/device/gpu/gpu_memory.h"

namespace tachyon::math {

size_t DetermineMsmDivisionsForMemory(size_t scalar_t_mem_size,
                                      size_t affine_t_mem_size,
                                      size_t projective_t_mem_size,
                                      size_t msm_size, size_t user_c,
                                      size_t bitsize, size_t precompute_factor,
                                      size_t batch_size) {
  size_t free_memory =
      device::gpu::GpuMemLimitInfo(device::gpu::MemoryUsage::kHigh);
  size_t shift = 0;
  uint32_t log_msm_size = base::bits::Log2Ceiling(msm_size);

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

}  // namespace tachyon::math
