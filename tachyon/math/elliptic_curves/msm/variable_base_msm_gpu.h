#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_

#include <memory>

#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm.h"

namespace tachyon::math {

template <typename GpuCurve>
class VariableBaseMSMGpu {
 public:
  using ScalarField = typename JacobianPoint<GpuCurve>::ScalarField;
  using CpuCurve = typename GpuCurve::CpuCurve;

  VariableBaseMSMGpu(gpuMemPool_t mem_pool, gpuStream_t stream)
      : impl_(std::make_unique<IcicleMSM<GpuCurve>>(mem_pool, stream)) {}
  VariableBaseMSMGpu(const VariableBaseMSMGpu& other) = delete;
  VariableBaseMSMGpu& operator=(const VariableBaseMSMGpu& other) = delete;

  bool Run(const device::gpu::GpuMemory<AffinePoint<GpuCurve>>& bases,
           const device::gpu::GpuMemory<ScalarField>& scalars, size_t size,
           JacobianPoint<CpuCurve>* cpu_result) {
    return impl_->Run(bases, scalars, size, cpu_result);
  }

 private:
  std::unique_ptr<IcicleMSM<GpuCurve>> impl_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_
