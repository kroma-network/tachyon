#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_KERNELS_CU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_KERNELS_CU_H_

#include "tachyon/device/gpu/cuda/cuda_memory.h"
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

}  // namespace tachyon::math::kernels

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_CUZK_KERNELS_CU_H_
