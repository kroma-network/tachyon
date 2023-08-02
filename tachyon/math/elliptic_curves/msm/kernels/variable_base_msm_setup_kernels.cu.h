#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_VARIABLE_BASE_MSM_SETUP_KERNELS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_VARIABLE_BASE_MSM_SETUP_KERNELS_H_

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"

namespace tachyon::math::kernels {
namespace msm {
using namespace device::gpu;

#define MAX_THREADS 128

template <typename Curve,
          typename BaseField = typename PointXYZZ<Curve>::BaseField>
__global__ void InitializeBucketsKernel(PointXYZZ<Curve>* buckets,
                                        unsigned int count) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  constexpr size_t kUint4Count = sizeof(BaseField) / sizeof(uint4);
  unsigned int bucket_index = gid / kUint4Count;
  unsigned int element_index = gid % kUint4Count;
  uint4* elements = const_cast<uint4*>(
      reinterpret_cast<const uint4*>(&buckets[bucket_index].zz()));
  Store<uint4, CacheOperator::kStreaming>(elements + element_index, uint4());
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
  gpuError_t error = gpuGetLastError();
  GPU_LOG_IF(ERROR, error != gpuSuccess, error)
      << "Failed to InitializeBucketsKernel()";
  return error;
}
#undef MAX_THREADS

#define MAX_THREADS 128
template <typename ScalarField>
__global__ void ComputeBucketIndexesKernel(
    const ScalarField* __restrict__ scalars, unsigned int windows_count,
    unsigned int window_bits, unsigned int* __restrict__ bucket_indexes,
    unsigned int* __restrict__ base_indexes, unsigned int count) {
  constexpr unsigned kHighestBitMask = 0x80000000;
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
    unsigned int bucket_index = ScalarField::ExtractBits(
        scalar, window_index * window_bits, window_bits);
    bucket_index += borrow;
    borrow = 0;
    unsigned int sign = global_sign;
    if (bucket_index > top_bucket_index) {
      bucket_index = (top_bucket_index << 1) - bucket_index;
      borrow = 1;
      sign ^= kHighestBitMask;
    }
    bool is_top_window = window_index == windows_count - 1;
    unsigned int zero_mask = bucket_index ? 0 : kHighestBitMask;
    bucket_index =
        ((bucket_index &
          ((is_top_window ? top_window_top_bucket_index : top_bucket_index) -
           1))
         << 1) |
        (bucket_index >>
         (is_top_window ? top_window_signed_window_bits : signed_window_bits));
    unsigned int bucket_index_offset =
        is_top_window
            ? (scalar_index & top_window_unused_mask) << top_window_used_bits
            : 0;
    unsigned int output_index = window_index * count + scalar_index;
    bucket_indexes[output_index] =
        zero_mask | window_mask | bucket_index_offset | bucket_index;
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
  gpuError_t error = gpuGetLastError();
  GPU_LOG_IF(ERROR, error != gpuSuccess, error)
      << "Failed to ComputeBucketIndexesKernel()";
  return error;
}
#undef MAX_THREADS

#define MAX_THREADS 128
__global__ void RemoveZeroBucketsKernel(
    const unsigned int* unique_bucket_indexes, unsigned int* bucket_run_lengths,
    const unsigned int* bucket_runs_count, const unsigned int count) {
  constexpr unsigned int kHighestBitMask = 0x80000000;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  unsigned int runs_count = *bucket_runs_count;
  unsigned int bucket_index = unique_bucket_indexes[gid];
  bool is_zero = bucket_index & kHighestBitMask;
  if (gid >= runs_count || is_zero) bucket_run_lengths[gid] = 0;
}

gpuError_t RemoveZeroBuckets(unsigned int* unique_bucket_indexes,
                             unsigned int* bucket_run_lengths,
                             const unsigned int* bucket_runs_count,
                             const unsigned int count, gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  RemoveZeroBucketsKernel<<<grid_dim, block_dim, 0, stream>>>(
      unique_bucket_indexes, bucket_run_lengths, bucket_runs_count, count);
  gpuError_t error = gpuGetLastError();
  GPU_LOG_IF(ERROR, error != gpuSuccess, error)
      << "Failed to RemoveZeroBucketsKernel()";
  return error;
}
#undef MAX_THREADS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
template <typename Curve, bool IsFirst>
__global__ void AggregateBucketsKernel(
    unsigned int* __restrict__ base_indexes,
    unsigned int* __restrict__ bucket_run_offsets,
    unsigned int* __restrict__ bucket_run_lengths,
    unsigned int* __restrict__ bucket_indexes,
    const AffinePoint<Curve>* __restrict__ bases,
    PointXYZZ<Curve>* __restrict__ buckets, unsigned int count) {
  constexpr unsigned int kNegativeSign = 0x80000000;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  unsigned int length = bucket_run_lengths[gid];
  if (length == 0) return;
  unsigned int base_indexes_offset = bucket_run_offsets[gid];
  unsigned int* indexes = base_indexes + base_indexes_offset;
  unsigned int bucket_index = bucket_indexes[gid] >> 1;
  PointXYZZ<Curve> bucket;
  if constexpr (IsFirst) {
    unsigned int base_index = *indexes++;
    unsigned int sign = base_index & kNegativeSign;
    base_index &= ~kNegativeSign;
    auto base =
        Load<AffinePoint<Curve>, CacheOperator::kNone>(bases + base_index);
    if (sign) {
      base = base.NegInPlace();
    }
    bucket = base.ToXYZZ();
  } else {
    bucket = Load<PointXYZZ<Curve>, CacheOperator::kStreaming>(buckets +
                                                               bucket_index);
  }
  for (unsigned int i = IsFirst ? 1 : 0; i < length; i++) {
    unsigned int base_index = *indexes++;
    unsigned int sign = base_index & kNegativeSign;
    base_index &= ~kNegativeSign;
    auto base =
        Load<AffinePoint<Curve>, CacheOperator::kNone>(bases + base_index);
    if (sign) {
      base.NegInPlace();
    }
    bucket += base;
  }
  Store<PointXYZZ<Curve>, CacheOperator::kStreaming>(buckets + bucket_index,
                                                     bucket);
}

template <typename Curve>
gpuError_t AggregateBuckets(const bool is_first, unsigned int* base_indexes,
                            unsigned int* bucket_run_offsets,
                            unsigned int* bucket_run_lengths,
                            unsigned int* bucket_indexes,
                            const AffinePoint<Curve>* bases,
                            PointXYZZ<Curve>* buckets, const unsigned count,
                            gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  auto kernel =
      is_first ? AggregateBucketsKernel<Curve, true> : AggregateBucketsKernel<Curve, false>;
  kernel<<<grid_dim, block_dim, 0, stream>>>(base_indexes, bucket_run_offsets,
                                             bucket_run_lengths, bucket_indexes,
                                             bases, buckets, count);
  gpuError_t error = gpuGetLastError();
  GPU_LOG_IF(ERROR, error != gpuSuccess, error)
      << "Failed to AggregateBucketsKernel()";
  return error;
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
  gpuError_t error = gpuGetLastError();
  GPU_LOG_IF(ERROR, error != gpuSuccess, error)
      << "Failed to ExtractTopBucketsKernel()";
  return error;
}
#undef MAX_THREADS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
template <typename Curve>
__global__ void SplitWindowsKernel(
    unsigned int source_window_bits_count, unsigned int source_windows_count,
    const PointXYZZ<Curve>* __restrict__ source_buckets,
    PointXYZZ<Curve>* __restrict__ target_buckets, unsigned int count) {
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
        Load<PointXYZZ<Curve>, CacheOperator::kNone>(source_buckets +
                                                     load_offset);
    target_bucket = i == target_partition_index ? source_bucket
                                                : target_bucket + source_bucket;
  }
  Store<PointXYZZ<Curve>, CacheOperator::kStreaming>(target_buckets + gid,
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
  gpuError_t error = gpuGetLastError();
  GPU_LOG_IF(ERROR, error != gpuSuccess, error)
      << "Failed to SplitWindowsKernel()";
  return error;
}
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
template <typename Curve>
__global__ void ReduceBucketsKernel(PointXYZZ<Curve>* buckets,
                                    unsigned int count) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  buckets += gid;
  auto a = Load<PointXYZZ<Curve>, CacheOperator::kNone>(buckets);
  auto b = Load<PointXYZZ<Curve>, CacheOperator::kNone>(buckets + count);
  Store<PointXYZZ<Curve>, CacheOperator::kStreaming>(buckets, a + b);
}

template <typename Curve>
gpuError_t ReduceBuckets(PointXYZZ<Curve>* buckets, unsigned int count,
                         gpuStream_t stream) {
  dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ReduceBucketsKernel<<<grid_dim, block_dim, 0, stream>>>(buckets, count);
  gpuError_t error = gpuGetLastError();
  GPU_LOG_IF(ERROR, error != gpuSuccess, error)
      << "Failed to ReduceBucketsKernel()";
  return error;
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
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  unsigned int signed_bits_count_pass_one = bits_count_pass_one - 1;
  unsigned int window_index = gid / bits_count_pass_one;
  unsigned int window_tid = gid % bits_count_pass_one;
  PointXYZZ<Curve> pz;
  if (window_tid == signed_bits_count_pass_one || gid == count - 1) {
    pz = Load<PointXYZZ<Curve>, CacheOperator::kNone>(top_buckets +
                                                      window_index);
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
    pz = Load<PointXYZZ<Curve>, CacheOperator::kNone>(source + sid);
  }
  Store<JacobianPoint<Curve>, CacheOperator::kStreaming>(target + gid,
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
  gpuError_t error = gpuGetLastError();
  GPU_LOG_IF(ERROR, error != gpuSuccess, error)
      << "Failed to LastPassGatherKernel()";
  return error;
}
#undef MAX_THREADS

template <typename T>
bool SetKernelAttributes(T* func) {
  gpuError_t error = cudaFuncSetCacheConfig(func, cudaFuncCachePreferL1);
  GPU_CHECK(error == gpuSuccess, error) << "Failed to cudaFuncSetCacheConfig()";
  error =
      cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout,
                           cudaSharedmemCarveoutMaxL1);
  GPU_CHECK(error == gpuSuccess, error) << "Failed to cudaFuncSetAttribute()";
  return true;
}

template <typename Curve, typename ScalarField = typename Curve::ScalarField>
bool SetupKernels() {
  if (!SetKernelAttributes(InitializeBucketsKernel<Curve>)) return false;
  if (!SetKernelAttributes(ComputeBucketIndexesKernel<ScalarField>))
    return false;
  if (!SetKernelAttributes(RemoveZeroBucketsKernel)) return false;
  if (!SetKernelAttributes(AggregateBucketsKernel<Curve, false>)) return false;
  if (!SetKernelAttributes(AggregateBucketsKernel<Curve, true>)) return false;
  if (!SetKernelAttributes(ExtractTopBucketsKernel<Curve>)) return false;
  if (!SetKernelAttributes(SplitWindowsKernel<Curve>)) return false;
  if (!SetKernelAttributes(ReduceBucketsKernel<Curve>)) return false;
  if (!SetKernelAttributes(LastPassGatherKernel<Curve>)) return false;
  return true;
}

}  // namespace msm
}  // namespace tachyon::math::kernels

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_VARIABLE_BASE_MSM_SETUP_KERNELS_H_
