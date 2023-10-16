// Copyright cuZK authors.
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.cuzk and the LICENCE-APACHE.cuzk
// file.

#include "tachyon/math/elliptic_curves/msm/kernels/cuzk/cuzk_kernels.cu.h"

namespace tachyon::math::cuzk {

__global__ void ConvertELLToCSRTransposedStep1(CUZKELLSparseMatrix ell_matrix,
                                               CUZKCSRSparseMatrix csr_matrix,
                                               unsigned int* row_ptr_offsets) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int tnum = gridDim.x * blockDim.x;

  while (tid < ell_matrix.rows) {
    unsigned int offset = tid * ell_matrix.cols;
    for (unsigned int i = 0; i < ell_matrix.row_lengths[tid]; ++i) {
      row_ptr_offsets[offset + i] = atomicAdd(
          &csr_matrix.row_ptrs[ell_matrix.col_indices[offset + i] + 1], 1);
    }
    tid += tnum;
  }
}

__global__ void ConvertELLToCSRTransposedStep2(CUZKCSRSparseMatrix csr_matrix) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int tnum = gridDim.x * blockDim.x;

  unsigned int start = (csr_matrix.rows + tnum) / tnum * tid;
  unsigned int end = (csr_matrix.rows + tnum) / tnum * (tid + 1);
  for (unsigned int i = start + 1; i < end && i <= csr_matrix.rows; ++i) {
    csr_matrix.row_ptrs[i] += csr_matrix.row_ptrs[i - 1];
  }
}

__global__ void ConvertELLToCSRTransposedStep3(CUZKCSRSparseMatrix csr_matrix,
                                               unsigned int i,
                                               unsigned int stride) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned int idx =
      tid / (1 << i) * (1 << (i + 1)) + (1 << i) + tid % (1 << i);
  unsigned int widx = (idx + 1) * stride - 1;
  unsigned int ridx = (idx / (1 << i) * (1 << i)) * stride - 1;
  if (widx <= csr_matrix.rows) {
    csr_matrix.row_ptrs[widx] += csr_matrix.row_ptrs[ridx];
  }
}

__global__ void ConvertELLToCSRTransposedStep4(CUZKCSRSparseMatrix csr_matrix) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int tnum = gridDim.x * blockDim.x;

  unsigned int start = (csr_matrix.rows + tnum) / tnum * (tid + 1);
  unsigned int end = (csr_matrix.rows + tnum) / tnum * (tid + 2) - 1;
  for (unsigned int i = start; i < end && i <= csr_matrix.rows; ++i) {
    csr_matrix.row_ptrs[i] += csr_matrix.row_ptrs[start - 1];
  }
}

__global__ void ConvertELLToCSRTransposedStep5(CUZKELLSparseMatrix ell_matrix,
                                               CUZKCSRSparseMatrix csr_matrix,
                                               unsigned int* row_ptr_offsets) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int tnum = gridDim.x * blockDim.x;

  while (tid < ell_matrix.rows) {
    unsigned int offset = tid * ell_matrix.cols;
    for (unsigned int i = 0; i < ell_matrix.row_lengths[tid]; ++i) {
      unsigned int ridx = offset + i;
      unsigned int widx = row_ptr_offsets[ridx] +
                          csr_matrix.row_ptrs[ell_matrix.col_indices[ridx]];
      csr_matrix.col_datas[widx] = {tid, ridx};
    }
    tid += tnum;
  }
}

}  // namespace tachyon::math::cuzk
