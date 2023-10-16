// Copyright (c) 2022 Matter Labs
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.era-bellman-cuda and the
// LICENCE-APACHE.era-bellman-cuda file.

#include "tachyon/math/elliptic_curves/msm/kernels/bellman/bellman_msm_kernels.cu.h"

namespace tachyon::math::bellman {

#define MAX_THREADS 128
__global__ void RemoveZeroBucketsKernel(
    const unsigned int* unique_bucket_indexes, unsigned int* bucket_run_lengths,
    const unsigned int* bucket_runs_count, unsigned int count) {
  constexpr unsigned int kHighestBitMask = 0x80000000;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  unsigned int runs_count = *bucket_runs_count;
  unsigned int bucket_index = unique_bucket_indexes[gid];
  bool is_zero = bucket_index & kHighestBitMask;
  if (gid >= runs_count || is_zero) bucket_run_lengths[gid] = 0;
}

gpuError_t RemoveZeroBuckets(const unsigned int* unique_bucket_indexes,
                             unsigned int* bucket_run_lengths,
                             const unsigned int* bucket_runs_count,
                             unsigned int count, gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  RemoveZeroBucketsKernel<<<grid_dim, block_dim, 0, stream>>>(
      unique_bucket_indexes, bucket_run_lengths, bucket_runs_count, count);
  return LOG_IF_GPU_LAST_ERROR("Failed to RemoveZeroBucketsKernel()");
}
#undef MAX_THREADs

}  // namespace tachyon::math::bellman
