#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_

#include "tachyon/math/elliptic_curves/msm/algorithms/bellman/bellman_msm.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/cuzk/cuzk.h"

namespace tachyon::math {

template <typename GpuCurve>
class VariableBaseMSMGpu {
 public:
  using ScalarField = typename JacobianPoint<GpuCurve>::ScalarField;
  using CpuCurve = typename SWCurveTraits<GpuCurve>::CpuCurve;

  VariableBaseMSMGpu(MSMAlgorithmKind kind, gpuMemPool_t mem_pool,
                     gpuStream_t stream) {
    algo_ = Create(kind, mem_pool, stream);
  }
  VariableBaseMSMGpu(const VariableBaseMSMGpu& other) = delete;
  VariableBaseMSMGpu& operator=(const VariableBaseMSMGpu& other) = delete;

  bool Run(const device::gpu::GpuMemory<AffinePoint<GpuCurve>>& bases,
           const device::gpu::GpuMemory<ScalarField>& scalars, size_t size,
           JacobianPoint<CpuCurve>* cpu_result) {
    return algo_->Run(bases, scalars, size, cpu_result);
  }

 private:
  static std::unique_ptr<MSMGpuAlgorithm<GpuCurve>> Create(
      MSMAlgorithmKind kind, gpuMemPool_t mem_pool, gpuStream_t stream) {
    switch (kind) {
      case MSMAlgorithmKind::kBellmanMSM:
        return std::make_unique<BellmanMSM<GpuCurve>>(mem_pool, stream);
      case MSMAlgorithmKind::kCUZK:
        return std::make_unique<CUZK<GpuCurve>>(mem_pool, stream);
      case MSMAlgorithmKind::kPippenger:
        break;
    }
    NOTREACHED();
    return nullptr;
  }

  std::unique_ptr<MSMGpuAlgorithm<GpuCurve>> algo_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_
