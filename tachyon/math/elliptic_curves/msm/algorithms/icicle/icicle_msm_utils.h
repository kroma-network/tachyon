#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_UTILS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_UTILS_H_

#include <stddef.h>

#include "tachyon/export.h"

namespace tachyon::math {

// NOTE(GideokKim): The formula for memory usage estimation provided in the
// document did not match the actual memory allocation, so some of the formula
// was modified. |scalars_memory_size| and |points_memory_size| are exactly the
// same, and |scalar_indices_memory_size| internally uses the sort function of
// the cub library to set some free memory. |buckets_memory_size| uses more
// memory than the actual formula, so it was modified to an empirically more
// appropriate formula. See
// https://dev.ingonyama.com/icicle/primitives/msm#memory-usage-estimation
TACHYON_EXPORT size_t DetermineMsmDivisionsForMemory(
    size_t scalar_t_mem_size, size_t affine_t_mem_size,
    size_t projective_t_mem_size, size_t msm_size, size_t user_c,
    size_t bitsize, size_t precompute_factor, size_t batch_size);

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_ICICLE_ICICLE_MSM_UTILS_H_
