#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_H_

#include "gtest/gtest_prod.h"

#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/cuzk_csr_sparse_matrix.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/cuzk_ell_sparse_matrix.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger_base.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger_ctx.h"
#include "tachyon/math/elliptic_curves/msm/kernels/cuzk_kernels.cu.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/sw_curve_traits.h"

namespace tachyon::math {

template <typename GpuCurve>
class CUZK : public PippengerBase<AffinePoint<GpuCurve>> {
 public:
  using BaseField = typename AffinePoint<GpuCurve>::BaseField;
  using ScalarField = typename AffinePoint<GpuCurve>::ScalarField;
  using Bucket = typename PippengerBase<GpuCurve>::Bucket;

  using CpuCurve = typename SWCurveTraits<GpuCurve>::CpuCurve;

  CUZK() : CUZK(nullptr, nullptr) {}
  CUZK(gpuMemPool_t mem_pool, gpuStream_t stream)
      : mem_pool_(mem_pool), stream_(stream) {
    // TODO(chokobole): Why grid_size is fixed to 512??
    constexpr unsigned int kDefaultGridSize = 512;
    constexpr unsigned int kDefaultBlockSize = 32;
    grid_size_ = kDefaultGridSize;
    block_size_ = kDefaultBlockSize;
  }
  CUZK(const CUZK& other) = delete;
  CUZK& operator=(const CUZK& other) = delete;

  void SetContextForTesting(const PippengerCtx& ctx) { ctx_ = ctx; }

  void SetGroupsForTesting(unsigned int start_group, unsigned int end_group) {
    start_group_ = start_group;
    end_group_ = end_group;
  }

  void SetSizesForTesting(unsigned int grid_size, unsigned int block_size) {
    grid_size_ = grid_size;
    block_size_ = block_size;
  }

  bool Run(const device::gpu::GpuMemory<AffinePoint<GpuCurve>>& bases,
           const device::gpu::GpuMemory<ScalarField>& scalars,
           PointXYZZ<CpuCurve>* cpu_result) {
    if (bases.size() != scalars.size()) {
      LOG(ERROR) << "bases.size() and scalars.size() don't match";
      return false;
    }

    if (device_count_ == 0) {
      gpuError_t error = LOG_IF_GPU_ERROR(cudaGetDeviceCount(&device_count_),
                                          "Failed to cudaGetDeviceCount()");
      error = LOG_IF_GPU_ERROR(cudaGetDevice(&device_id_),
                               "Failed to cudaGetDevice()");
    }
    ctx_ = PippengerCtx::CreateDefault<ScalarField>(scalars.size());
    start_group_ =
        (ctx_.window_count + device_count_ - 1) / device_count_ * device_id_;
    end_group_ = (ctx_.window_count + device_count_ - 1) / device_count_ *
                 (device_id_ + 1);
    if (start_group_ > ctx_.window_count) start_group_ = ctx_.window_count;
    if (end_group_ > ctx_.window_count) end_group_ = ctx_.window_count;

    auto buckets = device::gpu::GpuMemory<PointXYZZ<GpuCurve>>::MallocManaged(
        ctx_.GetWindowLength() * (end_group_ - start_group_));
    if (!CreateBuckets(bases, scalars, buckets)) {
      return false;
    }

    NOTIMPLEMENTED();

    gpuError_t error = gpuDeviceSynchronize();
    return error == gpuSuccess;
  }

 private:
  FRIEND_TEST(CUZKTest, ReduceBuckets);

  bool CreateBuckets(
      const device::gpu::GpuMemory<AffinePoint<GpuCurve>>& bases,
      const device::gpu::GpuMemory<ScalarField>& scalars,
      device::gpu::GpuMemory<PointXYZZ<GpuCurve>>& buckets) const {
    unsigned int thread_num = grid_size_ * block_size_;

    CUZKELLSparseMatrix ell_matrix;
    ell_matrix.rows = thread_num;
    ell_matrix.cols = (scalars.size() + thread_num - 1) / thread_num;
    auto row_lengths =
        device::gpu::GpuMemory<unsigned int>::MallocManaged(ell_matrix.rows);
    auto col_indicies = device::gpu::GpuMemory<unsigned int>::MallocManaged(
        ell_matrix.rows * ell_matrix.cols);
    ell_matrix.row_lengths = row_lengths.get();
    ell_matrix.col_indices = col_indicies.get();

    CUZKCSRSparseMatrix csr_matrix_transposed;
    csr_matrix_transposed.rows = ctx_.GetWindowLength();
    csr_matrix_transposed.cols = thread_num;
    auto row_ptrs = device::gpu::GpuMemory<unsigned int>::MallocManaged(
        csr_matrix_transposed.rows + 1);
    auto col_datas =
        device::gpu::GpuMemory<CUZKCSRSparseMatrix::Element>::MallocManaged(
            scalars.size());
    csr_matrix_transposed.row_ptrs = row_ptrs.get();
    csr_matrix_transposed.col_datas = col_datas.get();
    csr_matrix_transposed.col_datas_size = col_datas.size();

    auto row_ptr_offsets = device::gpu::GpuMemory<unsigned int>::MallocManaged(
        col_indicies.size() + 1);
    for (unsigned int i = start_group_; i < end_group_; ++i) {
      row_lengths.Memset();
      kernels::WriteBucketIndexesToELLMatrix<<<grid_size_, block_size_>>>(
          ctx_, i, scalars.get(), ell_matrix);
      gpuError_t error = gpuGetLastError();
      if (error != gpuSuccess) {
        LOG(ERROR) << "Failed to kernels::WriteBucketIndexesToELLMatrix()";
        return false;
      }

      row_ptrs.Memset();
      row_ptr_offsets.Memset();
      // TODO(chokobole): Can we remove this?
      col_datas.Memset();
      if (!ConvertELLToCSRTransposed(ell_matrix, csr_matrix_transposed, i,
                                     row_ptr_offsets.get())) {
        return false;
      }

      NOTIMPLEMENTED();
    }
    return true;
  }

  bool ConvertELLToCSRTransposed(const CUZKELLSparseMatrix& ell_matrix,
                                 CUZKCSRSparseMatrix& csr_matrix,
                                 unsigned int idx,
                                 unsigned int* row_ptr_offsets) const {
    kernels::ConvertELLToCSRTransposedStep1<<<grid_size_, block_size_>>>(
        ell_matrix, csr_matrix, row_ptr_offsets);
    gpuError_t error = gpuGetLastError();
    if (error != gpuSuccess) {
      LOG(ERROR) << "Failed to kernels::ConvertELLToCSRTransposedStep1()";
      return false;
    }
    gpuStreamSynchronize(0);

    kernels::ConvertELLToCSRTransposedStep2<<<grid_size_, block_size_>>>(
        csr_matrix);
    error = gpuGetLastError();
    if (error != gpuSuccess) {
      LOG(ERROR) << "Failed to kernels::ConvertELLToCSRTransposedStep2()";
      return false;
    }
    gpuStreamSynchronize(0);

    unsigned int thread_num = grid_size_ * block_size_;
    unsigned int row_ptrs_size = csr_matrix.rows + 1;
    unsigned int stride = (row_ptrs_size + thread_num - 1) / thread_num;

    for (unsigned int i = 0; i < log2(thread_num); ++i) {
      kernels::ConvertELLToCSRTransposedStep3<<<grid_size_ / 2, block_size_>>>(
          csr_matrix, i, stride);
      error = gpuGetLastError();
      if (error != gpuSuccess) {
        LOG(ERROR) << "Failed to kernels::ConvertELLToCSRTransposedStep3()";
        return false;
      }
      gpuStreamSynchronize(0);
    }

    kernels::ConvertELLToCSRTransposedStep4<<<grid_size_, block_size_>>>(
        csr_matrix);
    error = gpuGetLastError();
    if (error != gpuSuccess) {
      LOG(ERROR) << "Failed to kernels::ConvertELLToCSRTransposedStep4()";
      return false;
    }
    gpuStreamSynchronize(0);

    kernels::ConvertELLToCSRTransposedStep5<<<grid_size_, block_size_>>>(
        ell_matrix, csr_matrix, row_ptr_offsets);
    error = gpuGetLastError();
    if (error != gpuSuccess) {
      LOG(ERROR) << "Failed to kernels::ConvertELLToCSRTransposedStep5()";
      return false;
    }
    gpuStreamSynchronize(0);

    return true;
  }

  bool ReduceBuckets(
      device::gpu::GpuMemory<PointXYZZ<GpuCurve>> buckets,
      device::gpu::GpuMemory<PointXYZZ<GpuCurve>>& result) const {
    unsigned int group_grid = 1;
    while (group_grid * 2 < grid_size_ / (end_group_ - start_group_)) {
      group_grid *= 2;
    }

    unsigned int gnum = group_grid * block_size_;
    auto intermediate_results =
        device::gpu::GpuMemory<PointXYZZ<GpuCurve>>::MallocManaged(
            (end_group_ - start_group_) * gnum);

    kernels::ReduceBucketsStep1<<<group_grid*(end_group_ - start_group_),
                                  block_size_>>>(
        ctx_, buckets.get(), intermediate_results.get(), group_grid);
    gpuError_t error = gpuGetLastError();
    if (error != gpuSuccess) {
      LOG(ERROR) << "Failed to kernels::ReduceBucketsStep1()";
      return false;
    }
    gpuStreamSynchronize(0);

    unsigned int t_count = gnum;
    unsigned int count = 1;
    while (t_count != 1) {
      kernels::ReduceBucketsStep2<<<group_grid*(end_group_ - start_group_),
                                    block_size_>>>(intermediate_results.get(),
                                                   group_grid, count);
      gpuError_t error = gpuGetLastError();
      if (error != gpuSuccess) {
        LOG(ERROR) << "Failed to kernels::ReduceBucketsStep2()";
        return false;
      }
      gpuStreamSynchronize(0);

      t_count = (t_count + 1) / 2;
      count *= 2;
    }

    kernels::ReduceBucketsStep3<<<1, 1>>>(ctx_, intermediate_results.get(),
                                          start_group_, end_group_, gnum,
                                          result.get());
    error = gpuGetLastError();
    if (error != gpuSuccess) {
      LOG(ERROR) << "Failed to kernels::ReduceBucketsStep3()";
      return false;
    }
    return true;
  }

  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;

  PippengerCtx ctx_;
  int device_count_ = 0;
  int device_id_ = 0;
  unsigned int grid_size_ = 0;
  unsigned int block_size_ = 0;
  unsigned int start_group_ = 0;
  unsigned int end_group_ = 0;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_H_
