#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_KERNELS_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_KERNELS_CU_H_

#include "tachyon/device/gpu/cuda/cuda_memory.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/cuzk_csr_sparse_matrix.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/cuzk_ell_sparse_matrix.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger_ctx.h"
#include "tachyon/math/elliptic_curves/point_xyzz.h"

namespace tachyon::math::kernels {

// This is a pStoreECPoints in the Algorithm3 section from
// https://eprint.iacr.org/2022/1321.pdf
template <typename ScalarField>
__global__ void WriteBucketIndexesToELLMatrix(PippengerCtx ctx,
                                              unsigned int window_index,
                                              const ScalarField* scalars,
                                              CUZKELLSparseMatrix matrix) {
  using namespace device::gpu;
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int tnum = gridDim.x * blockDim.x;

  unsigned int start = (ctx.size + tnum - 1) / tnum * tid;
  unsigned int end = (ctx.size + tnum - 1) / tnum * (tid + 1);
  for (unsigned int i = start; i < end && i < ctx.size; ++i) {
    ScalarField scalar = ScalarField::FromMontgomery(
        Load<ScalarField, CacheOperator::kNone>(&scalars[i]).ToBigInt());
    // TODO(chokobole): Replace this with BigInt<N>.ExtractBits32().
    unsigned int bucket_index =
        scalar.ExtractBits(window_index * ctx.window_bits, ctx.window_bits);
    matrix.Insert(tid, bucket_index);
  }
}

// This is a combination of pELL2CSR and pTranspose in the Algorithm 3 section
// from https://eprint.iacr.org/2022/1321.pdf
__global__ void ConvertELLToCSRTransposedStep1(CUZKELLSparseMatrix ell_matrix,
                                               CUZKCSRSparseMatrix csr_matrix,
                                               unsigned int* row_ptr_offsets);

__global__ void ConvertELLToCSRTransposedStep2(CUZKCSRSparseMatrix csr_matrix);

__global__ void ConvertELLToCSRTransposedStep3(CUZKCSRSparseMatrix csr_matrix,
                                               unsigned int i,
                                               unsigned int stride);

__global__ void ConvertELLToCSRTransposedStep4(CUZKCSRSparseMatrix csr_matrix);

__global__ void ConvertELLToCSRTransposedStep5(CUZKELLSparseMatrix ell_matrix,
                                               CUZKCSRSparseMatrix csr_matrix,
                                               unsigned int* row_ptr_offsets);

// This is pBucketPointsReduction in the Algorithm 3 and the Algorithm 4 section
// from https://eprint.iacr.org/2022/1321.pdf
template <typename Curve>
__global__ void ReduceBucketsStep1(
    PippengerCtx ctx, PointXYZZ<Curve>* __restrict__ buckets,
    PointXYZZ<Curve>* __restrict__ intermediate_results,
    unsigned int group_grid) {
  unsigned int gid = blockIdx.x / group_grid;
  unsigned int gnum = group_grid * blockDim.x;
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int gtid = tid % gnum;

  unsigned int total = ctx.GetWindowLength();
  unsigned int start = (total + gnum - 1) / gnum * gtid;
  unsigned int end = (total + gnum - 1) / gnum * (gtid + 1);

  PointXYZZ<Curve> result = PointXYZZ<Curve>::Zero();
  PointXYZZ<Curve> running_sum = PointXYZZ<Curve>::Zero();
  for (unsigned int i = end > total ? total - 1 : end - 1; i >= start && i > 0;
       --i) {
    running_sum += buckets[gid * total + i];
    result += running_sum;
  }

  if (start != 0) {
    result -= running_sum;
    result += running_sum.ScalarMul(BigInt<1>(start));
  }

  intermediate_results[gid * gnum + gtid] = result;
}

template <typename Curve>
__global__ void ReduceBucketsStep2(
    PointXYZZ<Curve>* __restrict__ intermediate_results,
    unsigned int group_grid, unsigned int count) {
  unsigned int gid = blockIdx.x / group_grid;
  unsigned int gnum = group_grid * blockDim.x;
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int gtid = tid % gnum;

  if (gtid % (count * 2) == 0) {
    if (gtid + count < group_grid * blockDim.x) {
      intermediate_results[gid * gnum + gtid] +=
          intermediate_results[gid * gnum + gtid + count];
    }
  }
}

template <typename Curve>
__global__ void ReduceBucketsStep3(
    PippengerCtx ctx, PointXYZZ<Curve>* __restrict__ intermediate_results,
    unsigned int start_group, unsigned int end_group, unsigned int gnum,
    PointXYZZ<Curve>* result) {
  *result = PointXYZZ<Curve>::Zero();
  for (unsigned int k = ctx.window_count - 1; k <= ctx.window_count; --k) {
    if (k >= start_group && k < end_group) {
      for (unsigned int i = 0; i < ctx.window_bits; ++i) {
        result->DoubleInPlace();
      }
      result->AddInPlace(intermediate_results[(k - start_group) * gnum]);
    }
  }
}

}  // namespace tachyon::math::kernels

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_KERNELS_CU_H_
