#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_BELLMAN_BELLMAN_MSM_KERNELS_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_BELLMAN_BELLMAN_MSM_KERNELS_CU_H_

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"

namespace tachyon::math::bellman {

#define MAX_THREADS 128

template <typename Curve,
          typename BaseField = typename PointXYZZ<Curve>::BaseField>
__global__ void InitializeBucketsKernel(PointXYZZ<Curve>* buckets,
                                        unsigned int count) {
  using namespace device::gpu;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  constexpr size_t kUint4Count = sizeof(BaseField) / sizeof(uint4);
  unsigned int bucket_index = gid / kUint4Count;
  unsigned int element_index = gid % kUint4Count;
  // | x     | y     |
  // |-------| ------|
  // | 0 | 1 | 0 | 1 |
  // |-------| ------|
  // | zz    | zzz   |
  // |-------| ------|
  // | 0 | 1 | 0 | 1 |
  // <-- 0 -->
  // This sets zeros to above a certain region, and this is enough to make a
  // point in |buckets| zero.
  uint4* elements = const_cast<uint4*>(
      reinterpret_cast<const uint4*>(&buckets[bucket_index].zz()));
  Store<uint4, CacheOperator::kStreaming>(&elements[element_index], uint4());
}

template <typename Curve,
          typename BaseField = typename PointXYZZ<Curve>::BaseField>
gpuError_t InitializeBuckets(PointXYZZ<Curve>* buckets, unsigned int count,
                             gpuStream_t stream) {
  constexpr size_t kUint4Count = sizeof(BaseField) / sizeof(uint4);
  unsigned int count_u4 = kUint4Count * count;
  dim3 block_dim = count_u4 < MAX_THREADS ? count_u4 : MAX_THREADS;
  dim3 grid_dim = (count_u4 - 1) / block_dim.x + 1;
  InitializeBucketsKernel<<<grid_dim, block_dim, 0, stream>>>(buckets,
                                                              count_u4);
  return LOG_IF_GPU_LAST_ERROR("Failed to InitializeBucketsKernel()");
}
#undef MAX_THREADS

#define MAX_THREADS 128
template <typename ScalarField>
__global__ void ComputeBucketIndexesKernel(
    const ScalarField* __restrict__ scalars, unsigned int windows_count,
    unsigned int window_bits, unsigned int* __restrict__ bucket_indexes,
    unsigned int* __restrict__ base_indexes, unsigned int count) {
  using namespace device::gpu;
  constexpr unsigned int kHighestBitMask = 0x80000000;
  unsigned int scalar_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar_index >= count) return;
  unsigned int top_window_unused_bits =
      windows_count * window_bits - ScalarField::Config::kModulusBits;
  unsigned int top_window_unused_mask = (1 << top_window_unused_bits) - 1;
  unsigned int top_window_used_bits = window_bits - top_window_unused_bits;
  unsigned int signed_window_bits = window_bits - 1;
  unsigned int top_window_signed_window_bits = top_window_used_bits - 1;
  unsigned int top_bucket_index = 1 << signed_window_bits;
  unsigned int top_window_top_bucket_index = 1 << top_window_signed_window_bits;
  ScalarField pos_scalar = ScalarField::FromMontgomery(
      Load<ScalarField, CacheOperator::kNone>(&scalars[scalar_index])
          .ToBigInt());
  ScalarField neg_scalar = -pos_scalar;
  unsigned int global_sign = pos_scalar < neg_scalar ? 0 : kHighestBitMask;
  ScalarField scalar = global_sign ? neg_scalar : pos_scalar;
  unsigned int borrow = 0;
  for (unsigned int i = 0; i < windows_count; i++) {
    unsigned int window_index = i;
    unsigned int window_mask = window_index << window_bits;
    // TODO(chokobole): Replace this with BigInt<N>.ExtractBits32().
    unsigned int bucket_index =
        scalar.ExtractBits(window_index * window_bits, window_bits);
    bucket_index += borrow;
    borrow = 0;
    unsigned int sign = global_sign;
    // NOTE(chokobole): The code below should not be executed when dealing with
    // the top window. For instance, in the case of the curve bn254, where the
    // modulus uses only 254 bits, the |window_bits| amount to 16 bits when the
    // degree is 20.
    // Consequently, scalar decomposition appears as follows:
    //
    // S = S_0 + S_1 * 2^16 + S_2 * 2^32 + ... + S_15 * 2^240
    //
    // Even with the addition of a borrow to S_15, the resulting value remains
    // below 2^14 - 1. This ensures that it remains within the bounds of
    // |top_bucket_index|, which is 2^15.
    if (bucket_index > top_bucket_index) {
      // When |window_bits| is 4 bits, the value of |top_bucket_index| is 8. In
      // a scenario where |bucket_index| surpasses 8(let's assume it's 10), the
      // resulting |bucket_index| transformation is as follows:

      // 10 * B = 10 * B - 16 * B + 16 * B
      //        = -6 * B + 16 * B
      //        = 6 * (-B) + 16 * B
      bucket_index = (top_bucket_index << 1) - bucket_index;
      borrow = 1;
      sign ^= kHighestBitMask;
    }
    bool is_top_window = window_index == windows_count - 1;
    // Ultimately, a dot product is required between the scalar and the base:
    //
    // (S_0, S_1 * 2^16, S_2 * 2^32, ..., S_15 * 2^240) * (B, B, B, ..., B)
    //
    // When |bucket_index| is zero, it indicates the potential to optimize the
    // computation of the dot product by subsequently reducing the required
    // number of multiplications.
    // See RemoveZeroBucketsKernel() for more details.
    unsigned int zero_mask = bucket_index ? 0 : kHighestBitMask;
    // For bn254 scalar field with degree 20, |bucket_index| appears as follows:
    //
    // In case of top window,
    //
    // | 2      | 13          | 1    |
    // |--------|-------------|------|
    // | unused |bucket_index | sign |
    //
    // Otherwise,
    //
    // | 15           | 1    |
    // |--------------|------|
    // | bucket_index | sign |
    bucket_index =
        ((bucket_index &
          ((is_top_window ? top_window_top_bucket_index : top_bucket_index) -
           1))
         << 1) |
        (bucket_index >>
         (is_top_window ? top_window_signed_window_bits : signed_window_bits));
    // NOTE(chokobole): You can think of it as like padding.
    unsigned int bucket_index_offset =
        is_top_window
            ? (scalar_index & top_window_unused_mask) << top_window_used_bits
            : 0;
    unsigned int output_index = window_index * count + scalar_index;
    // | 1         | window_bits -1 | 31                       | 1    |
    // |-----------|----------------|--------------------------|------|
    // | zero_mask | window_mask    | bucket_index(w/ padding) | sign |
    bucket_indexes[output_index] =
        zero_mask | window_mask | bucket_index_offset | bucket_index;
    // | 1    | 31           |
    // |------|--------------|
    // | sign | scalar_index |
    base_indexes[output_index] = sign | scalar_index;
  }
}

template <typename ScalarField>
gpuError_t ComputeBucketIndexes(const ScalarField* scalars,
                                unsigned int windows_count,
                                unsigned int window_bits,
                                unsigned int* bucket_indexes,
                                unsigned int* base_indexes, unsigned int count,
                                gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ComputeBucketIndexesKernel<<<grid_dim, block_dim, 0, stream>>>(
      scalars, windows_count, window_bits, bucket_indexes, base_indexes, count);
  return LOG_IF_GPU_LAST_ERROR("Failed to ComputeBucketIndexesKernel()");
}
#undef MAX_THREADS

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
#undef MAX_THREADS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
template <typename Curve, bool IsFirst>
__global__ void AggregateBucketsKernel(
    const unsigned int* __restrict__ base_indexes,
    const unsigned int* __restrict__ bucket_run_offsets,
    const unsigned int* __restrict__ bucket_run_lengths,
    const unsigned int* __restrict__ bucket_indexes,
    const AffinePoint<Curve>* __restrict__ bases,
    PointXYZZ<Curve>* __restrict__ buckets, unsigned int count) {
  using namespace device::gpu;
  constexpr unsigned int kNegativeSign = 0x80000000;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  unsigned int length = bucket_run_lengths[gid];
  // NOTE(chokobole): This is zeroed out in RemoveZeroBucketsKernel().
  if (length == 0) return;
  unsigned int base_indexes_offset = bucket_run_offsets[gid];
  const unsigned int* indexes = &base_indexes[base_indexes_offset];
  // NOTE(chokobole): The last bit is sign bit.
  unsigned int bucket_index = bucket_indexes[gid] >> 1;
  PointXYZZ<Curve> bucket;
  if constexpr (IsFirst) {
    unsigned int base_index = *indexes++;
    unsigned int sign = base_index & kNegativeSign;
    base_index &= ~kNegativeSign;
    auto base =
        Load<AffinePoint<Curve>, CacheOperator::kNone>(&bases[base_index]);
    if (sign) {
      base = base.NegInPlace();
    }
    bucket = base.ToXYZZ();
  } else {
    bucket = Load<PointXYZZ<Curve>, CacheOperator::kStreaming>(
        &buckets[bucket_index]);
  }
  // What we need to compute is as follows:
  //
  // B_0 * S_0 + B_1 * S_1 + B_2 * S_2 + ... + B_{N - 1} * S_{N - 1}
  //
  // For each scalar, it's decomposed based on the window bits.
  // Let's take the example of the bn254 curve again:
  //
  // S_0 = S_0_0 + S_0_1 * 2^16 + S_0_2 * 2^32 + ... + S_0_15 * 2^240
  // S_1 = S_1_0 + S_1_1 * 2^16 + S_1_2 * 2^32 + ... + S_1_15 * 2^240
  // ...
  // S_{N - 1} = S_{N - 1}_0 + S_{N - 1}_1 * 2^16 + S_{N - 1}_2 * 2^32 + ... +
  // S_{N - 1}_15 * 2^240
  //
  // S_{i}_{j} ranges from 0 to 2^15. (Be cautious, it's not 2^16.)
  // The bucket index can be calculated as follows:
  //
  // bucket_index = j * 2^16 + S_{i}_{j}.
  //
  // This accumulates the base belonging to the same |bucket_index|.
  for (unsigned int i = IsFirst ? 1 : 0; i < length; i++) {
    unsigned int base_index = *indexes++;
    unsigned int sign = base_index & kNegativeSign;
    base_index &= ~kNegativeSign;
    auto base =
        Load<AffinePoint<Curve>, CacheOperator::kNone>(&bases[base_index]);
    if (sign) {
      base.NegInPlace();
    }
    bucket += base;
  }
  Store<PointXYZZ<Curve>, CacheOperator::kStreaming>(&buckets[bucket_index],
                                                     bucket);
}

template <typename Curve>
gpuError_t AggregateBuckets(bool is_first, const unsigned int* base_indexes,
                            const unsigned int* bucket_run_offsets,
                            const unsigned int* bucket_run_lengths,
                            const unsigned int* bucket_indexes,
                            const AffinePoint<Curve>* bases,
                            PointXYZZ<Curve>* buckets, unsigned int count,
                            gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  auto kernel = is_first ? AggregateBucketsKernel<Curve, true>
                         : AggregateBucketsKernel<Curve, false>;
  kernel<<<grid_dim, block_dim, 0, stream>>>(base_indexes, bucket_run_offsets,
                                             bucket_run_lengths, bucket_indexes,
                                             bases, buckets, count);
  return LOG_IF_GPU_LAST_ERROR("Failed to AggregateBucketsKernel()");
}

#define MAX_THREADS 32
template <typename Curve>
__global__ void ExtractTopBucketsKernel(PointXYZZ<Curve>* buckets,
                                        PointXYZZ<Curve>* top_buckets,
                                        unsigned int bits_count,
                                        unsigned int windows_count) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= windows_count) return;
  unsigned int bucket_index = gid << bits_count;
  top_buckets[gid] = buckets[bucket_index];
  buckets[bucket_index] = PointXYZZ<Curve>::Zero();
}

template <typename Curve>
gpuError_t ExtractTopBuckets(PointXYZZ<Curve>* buckets,
                             PointXYZZ<Curve>* top_buckets,
                             unsigned int bits_count,
                             unsigned int windows_count, gpuStream_t stream) {
  const dim3 block_dim =
      windows_count < MAX_THREADS ? windows_count : MAX_THREADS;
  const dim3 grid_dim = (windows_count - 1) / block_dim.x + 1;
  ExtractTopBucketsKernel<<<grid_dim, block_dim, 0, stream>>>(
      buckets, top_buckets, bits_count, windows_count);
  return LOG_IF_GPU_LAST_ERROR("Failed to ExtractTopBucketsKernel()");
}
#undef MAX_THREADS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
template <typename Curve>
__global__ void SplitWindowsKernel(
    unsigned int source_window_bits_count, unsigned int source_windows_count,
    const PointXYZZ<Curve>* __restrict__ source_buckets,
    PointXYZZ<Curve>* __restrict__ target_buckets, unsigned int count) {
  using namespace device::gpu;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  unsigned int target_window_bits_count = (source_window_bits_count + 1) >> 1;
  unsigned int target_windows_count = source_windows_count << 1;
  unsigned int target_partition_buckets_count = target_windows_count
                                                << target_window_bits_count;
  unsigned int target_partitions_count = count / target_partition_buckets_count;
  unsigned int target_partition_index = gid / target_partition_buckets_count;
  unsigned int target_partition_tid = gid % target_partition_buckets_count;
  unsigned int target_window_buckets_count = 1 << target_window_bits_count;
  unsigned int target_window_index =
      target_partition_tid / target_window_buckets_count;
  unsigned int target_window_tid =
      target_partition_tid % target_window_buckets_count;
  unsigned int split_index = target_window_index & 1;
  unsigned int source_window_buckets_per_target =
      source_window_bits_count & 1
          ? split_index
                ? (target_window_tid >> (target_window_bits_count - 1)
                       ? 0
                       : target_window_buckets_count)
                : 1 << (source_window_bits_count - target_window_bits_count)
          : target_window_buckets_count;
  unsigned int source_window_index = target_window_index >> 1;
  unsigned int source_offset = source_window_index << source_window_bits_count;
  unsigned int target_shift = target_window_bits_count * split_index;
  unsigned int target_offset = target_window_tid << target_shift;
  unsigned int global_offset = source_offset + target_offset;
  unsigned int index_mask = (1 << target_shift) - 1;
  PointXYZZ<Curve> target_bucket = PointXYZZ<Curve>::Zero();
  for (unsigned int i = target_partition_index;
       i < source_window_buckets_per_target; i += target_partitions_count) {
    unsigned int index_offset =
        i & index_mask | (i & ~index_mask) << target_window_bits_count;
    unsigned int load_offset = global_offset + index_offset;
    PointXYZZ<Curve> source_bucket =
        Load<PointXYZZ<Curve>, CacheOperator::kNone>(
            &source_buckets[load_offset]);
    target_bucket = i == target_partition_index ? source_bucket
                                                : target_bucket + source_bucket;
  }
  Store<PointXYZZ<Curve>, CacheOperator::kStreaming>(&target_buckets[gid],
                                                     target_bucket);
}

template <typename Curve>
gpuError_t SplitWindows(unsigned int source_window_bits_count,
                        unsigned int source_windows_count,
                        const PointXYZZ<Curve>* source_buckets,
                        PointXYZZ<Curve>* target_buckets, unsigned int count,
                        gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  SplitWindowsKernel<<<grid_dim, block_dim, 0, stream>>>(
      source_window_bits_count, source_windows_count, source_buckets,
      target_buckets, count);
  return LOG_IF_GPU_LAST_ERROR("Failed to SplitWindowsKernel()");
}
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
template <typename Curve>
__global__ void ReduceBucketsKernel(PointXYZZ<Curve>* buckets,
                                    unsigned int count) {
  using namespace device::gpu;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  buckets += gid;
  auto a = Load<PointXYZZ<Curve>, CacheOperator::kNone>(buckets);
  auto b = Load<PointXYZZ<Curve>, CacheOperator::kNone>(&buckets[count]);
  Store<PointXYZZ<Curve>, CacheOperator::kStreaming>(buckets, a + b);
}

template <typename Curve>
gpuError_t ReduceBuckets(PointXYZZ<Curve>* buckets, unsigned int count,
                         gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ReduceBucketsKernel<<<grid_dim, block_dim, 0, stream>>>(buckets, count);
  return LOG_IF_GPU_LAST_ERROR("Failed to ReduceBucketsKernel()");
}
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
template <typename Curve>
__global__ void LastPassGatherKernel(
    unsigned int bits_count_pass_one,
    const PointXYZZ<Curve>* __restrict__ source,
    const PointXYZZ<Curve>* top_buckets,
    JacobianPoint<Curve>* __restrict__ target, unsigned int count) {
  using namespace device::gpu;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  unsigned int signed_bits_count_pass_one = bits_count_pass_one - 1;
  unsigned int window_index = gid / bits_count_pass_one;
  unsigned int window_tid = gid % bits_count_pass_one;
  PointXYZZ<Curve> pz;
  if (window_tid == signed_bits_count_pass_one || gid == count - 1) {
    pz = Load<PointXYZZ<Curve>, CacheOperator::kNone>(
        &top_buckets[window_index]);
  } else {
    for (unsigned int bits_count = signed_bits_count_pass_one;
         bits_count > 1;) {
      bits_count = (bits_count + 1) >> 1;
      window_index <<= 1;
      if (window_tid >= bits_count) {
        window_index++;
        window_tid -= bits_count;
      }
    }
    unsigned int sid = (window_index << 1) + 1;
    pz = Load<PointXYZZ<Curve>, CacheOperator::kNone>(&source[sid]);
  }
  Store<JacobianPoint<Curve>, CacheOperator::kStreaming>(&target[gid],
                                                         pz.ToJacobian());
}

template <typename Curve>
gpuError_t LastPassGather(unsigned int bits_count_pass_one,
                          const PointXYZZ<Curve>* source,
                          const PointXYZZ<Curve>* top_buckets,
                          JacobianPoint<Curve>* target, unsigned int count,
                          gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  LastPassGatherKernel<<<grid_dim, block_dim, 0, stream>>>(
      bits_count_pass_one, source, top_buckets, target, count);
  return LOG_IF_GPU_LAST_ERROR("Failed to LastPassGatherKernel()");
}
#undef MAX_THREADS

template <typename T>
void SetKernelAttributes(T* func) {
  GPU_MUST_SUCCESS(cudaFuncSetCacheConfig(func, cudaFuncCachePreferL1),
                   "Failed to cudaFuncSetCacheConfig()");
  GPU_MUST_SUCCESS(
      cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout,
                           cudaSharedmemCarveoutMaxL1),
      "Failed to cudaFuncSetAttribute()");
}

template <typename Curve, typename ScalarField = typename Curve::ScalarField>
void SetupKernels() {
  SetKernelAttributes(InitializeBucketsKernel<Curve>);
  SetKernelAttributes(ComputeBucketIndexesKernel<ScalarField>);
  SetKernelAttributes(RemoveZeroBucketsKernel);
  SetKernelAttributes(AggregateBucketsKernel<Curve, false>);
  SetKernelAttributes(AggregateBucketsKernel<Curve, true>);
  SetKernelAttributes(ExtractTopBucketsKernel<Curve>);
  SetKernelAttributes(SplitWindowsKernel<Curve>);
  SetKernelAttributes(ReduceBucketsKernel<Curve>);
  SetKernelAttributes(LastPassGatherKernel<Curve>);
}

}  // namespace tachyon::math::bellman

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_BELLMAN_BELLMAN_MSM_KERNELS_CU_H_
