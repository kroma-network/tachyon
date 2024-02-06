// Copyright cuZK authors.
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.cuzk and the LICENCE-APACHE.cuzk
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_BN254_CUZK_KERNELS_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_BN254_CUZK_KERNELS_CU_H_

#include "tachyon/math/elliptic_curves/bn/bn254/g1_gpu.h"
#include "tachyon/math/elliptic_curves/msm/kernels/cuzk/cuzk_kernels.cu.h"

namespace tachyon::math::cuzk {

extern template __global__ void WriteBucketIndexesToELLMatrix<bn254::FrGpu>(
    MSMCtx ctx, unsigned int window_index, const bn254::FrGpu* scalars,
    CUZKELLSparseMatrix matrix);

extern template __global__ void
MultiplyCSRMatrixWithOneVectorStep1<bn254::G1CurveGpu>(
    MSMCtx ctx, unsigned int z, CUZKCSRSparseMatrix csr_matrix,
    const AffinePoint<bn254::G1CurveGpu>* bases,
    PointXYZZ<bn254::G1CurveGpu>* results, unsigned int bucket_index);

extern template __global__ void
MultiplyCSRMatrixWithOneVectorStep2<bn254::G1CurveGpu>(
    unsigned int start, unsigned int end, CUZKCSRSparseMatrix csr_matrix,
    const AffinePoint<bn254::G1CurveGpu>* bases,
    PointXYZZ<bn254::G1CurveGpu>* max_intermediate_results);

extern template __global__ void
MultiplyCSRMatrixWithOneVectorStep3<bn254::G1CurveGpu>(
    MSMCtx ctx, unsigned int index, unsigned int count,
    CUZKCSRSparseMatrix csr_matrix,
    const AffinePoint<bn254::G1CurveGpu>* __restrict__ bases,
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ max_intermediate_results,
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ results,
    unsigned int bucket_index);

extern template __global__ void
MultiplyCSRMatrixWithOneVectorStep4<bn254::G1CurveGpu>(
    unsigned int start, unsigned int end, unsigned int total, unsigned int i,
    unsigned int ptr, CUZKCSRSparseMatrix csr_matrix,
    const AffinePoint<bn254::G1CurveGpu>* __restrict__ bases,
    unsigned int* __restrict__ intermediate_indices,
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ intermediate_datas);

extern template __global__ void
MultiplyCSRMatrixWithOneVectorStep5<bn254::G1CurveGpu>(
    const unsigned int* __restrict__ intermediate_rows,
    const unsigned int* __restrict__ intermediate_indices,
    const PointXYZZ<bn254::G1CurveGpu>* __restrict__ intermediate_datas,
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ results, unsigned int total);

extern template __global__ void ReduceBucketsStep1<bn254::G1CurveGpu>(
    MSMCtx ctx, PointXYZZ<bn254::G1CurveGpu>* __restrict__ buckets,
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ intermediate_results,
    unsigned int group_grid);

extern template __global__ void ReduceBucketsStep2<bn254::G1CurveGpu>(
    PointXYZZ<bn254::G1CurveGpu>* __restrict__ intermediate_results,
    unsigned int group_grid, unsigned int count);

extern template __global__ void ReduceBucketsStep3<bn254::G1CurveGpu>(
    MSMCtx ctx, PointXYZZ<bn254::G1CurveGpu>* __restrict__ intermediate_results,
    unsigned int start_group, unsigned int end_group, unsigned int gnum,
    PointXYZZ<bn254::G1CurveGpu>* result);

}  // namespace tachyon::math::cuzk

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_BN254_CUZK_KERNELS_CU_H_
