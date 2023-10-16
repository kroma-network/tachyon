// Copyright (c) 2022 Matter Labs
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.era-bellman-cuda and the
// LICENCE-APACHE.era-bellman-cuda file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_BELLMAN_BN254_BELLMAN_MSM_KERNELS_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_BELLMAN_BN254_BELLMAN_MSM_KERNELS_CU_H_

#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/msm/kernels/bellman/bellman_msm_kernels.cu.h"

namespace tachyon::math::bellman {

extern template __global__ void InitializeBucketsKernel<bn254::G1CurveGpu>(
    PointXYZZ<bn254::G1CurveGpu>* buckets, unsigned int count);

extern template gpuError_t InitializeBuckets<bn254::G1CurveGpu>(
    PointXYZZ<bn254::G1CurveGpu>* buckets, unsigned int count,
    gpuStream_t stream);

extern template __global__ void ComputeBucketIndexesKernel<bn254::FrGpu>(
    const bn254::FrGpu* __restrict__ scalars, unsigned int windows_count,
    unsigned int window_bits, unsigned int* __restrict__ bucket_indexes,
    unsigned int* __restrict__ base_indexes, unsigned int count);

extern template gpuError_t ComputeBucketIndexes<bn254::FrGpu>(
    const bn254::FrGpu* scalars, unsigned int windows_count,
    unsigned int window_bits, unsigned int* bucket_indexes,
    unsigned int* base_indexes, unsigned int count, gpuStream_t stream);

extern template __global__ void
AggregateBucketsKernel<bn254::G1CurveGpu, false>(
    const unsigned int* __restrict__ base_indexes,
    const unsigned int* __restrict__ bucket_run_offsets,
    const unsigned int* __restrict__ bucket_run_lengths,
    const unsigned int* __restrict__ bucket_indexes,
    const AffinePoint<bn254::G1CurveGpu>* __restrict__ bases,
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ buckets, unsigned int count);

extern template __global__ void AggregateBucketsKernel<bn254::G1CurveGpu, true>(
    const unsigned int* __restrict__ base_indexes,
    const unsigned int* __restrict__ bucket_run_offsets,
    const unsigned int* __restrict__ bucket_run_lengths,
    const unsigned int* __restrict__ bucket_indexes,
    const AffinePoint<bn254::G1CurveGpu>* __restrict__ bases,
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ buckets, unsigned int count);

extern template gpuError_t AggregateBuckets<bn254::G1CurveGpu>(
    bool is_first, const unsigned int* base_indexes,
    const unsigned int* bucket_run_offsets,
    const unsigned int* bucket_run_lengths, const unsigned int* bucket_indexes,
    const AffinePoint<bn254::G1CurveGpu>* bases,
    PointXYZZ<bn254::G1CurveGpu>* buckets, unsigned int count,
    gpuStream_t stream);

extern template __global__ void ExtractTopBucketsKernel<bn254::G1CurveGpu>(
    PointXYZZ<bn254::G1CurveGpu>* buckets,
    PointXYZZ<bn254::G1CurveGpu>* top_buckets, unsigned int bits_count,
    unsigned int windows_count);

extern template gpuError_t ExtractTopBuckets<bn254::G1CurveGpu>(
    PointXYZZ<bn254::G1CurveGpu>* buckets,
    PointXYZZ<bn254::G1CurveGpu>* top_buckets, unsigned int bits_count,
    unsigned int windows_count, gpuStream_t stream);

extern template __global__ void SplitWindowsKernel<bn254::G1CurveGpu>(
    unsigned int source_window_bits_count, unsigned int source_windows_count,
    const PointXYZZ<bn254::G1CurveGpu>* __restrict__ source_buckets,
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ target_buckets,
    unsigned int count);

extern template gpuError_t SplitWindows<bn254::G1CurveGpu>(
    unsigned int source_window_bits_count, unsigned int source_windows_count,
    const PointXYZZ<bn254::G1CurveGpu>* source_buckets,
    PointXYZZ<bn254::G1CurveGpu>* target_buckets, unsigned int count,
    gpuStream_t stream);

extern template __global__ void ReduceBucketsKernel<bn254::G1CurveGpu>(
    PointXYZZ<bn254::G1CurveGpu>* buckets, unsigned int count);

extern template gpuError_t ReduceBuckets<bn254::G1CurveGpu>(
    PointXYZZ<bn254::G1CurveGpu>* buckets, unsigned int count,
    gpuStream_t stream);

extern template __global__ void LastPassGatherKernel<bn254::G1CurveGpu>(
    unsigned int bits_count_pass_one,
    const PointXYZZ<bn254::G1CurveGpu>* __restrict__ source,
    const PointXYZZ<bn254::G1CurveGpu>* top_buckets,
    JacobianPoint<bn254::G1CurveGpu>* __restrict__ target, unsigned int count);

extern template gpuError_t LastPassGather<bn254::G1CurveGpu>(
    unsigned int bits_count_pass_one,
    const PointXYZZ<bn254::G1CurveGpu>* source,
    const PointXYZZ<bn254::G1CurveGpu>* top_buckets,
    JacobianPoint<bn254::G1CurveGpu>* target, unsigned int count,
    gpuStream_t stream);

}  // namespace tachyon::math::bellman

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_BELLMAN_BN254_BELLMAN_MSM_KERNELS_CU_H_
