#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_CUDA_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_CUDA_H_

#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/math/elliptic_curves/msm/kernels/variable_base_msm_execution_kernels.cu.h"
#include "tachyon/math/elliptic_curves/msm/kernels/variable_base_msm_setup_kernels.cu.h"

namespace tachyon::math {

template <typename Curve>
class VariableBaseMSMCuda {
 public:
  static void Setup() { kernels::msm::SetupKernels<Curve>(); }

  static gpuError_t ExecuteAsync(
      const kernels::msm::ExecutionConfig<Curve>& config) {
    return kernels::msm::ExecuteAsync(config);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_CUDA_H_
