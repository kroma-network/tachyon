#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_CUDA_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_CUDA_H_

#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/math/elliptic_curves/msm/kernels/variable_base_msm_execution_kernels.cu.h"
#include "tachyon/math/elliptic_curves/msm/kernels/variable_base_msm_setup_kernels.cu.h"

namespace tachyon::math {

template <typename Curve>
class VariableBaseMSMCuda {
 public:
  using Config = typename Curve::Config;
  using ScalarField = typename JacobianPoint<Curve>::ScalarField;

  constexpr static size_t kModulusBits = ScalarField::Config::kModulusBits;

  static void Setup() { kernels::msm::SetupKernels<Curve>(); }

  static gpuError_t ExecuteAsync(
      const kernels::msm::ExecutionConfig<Curve>& config) {
    return kernels::msm::ExecuteAsync(config);
  }

  template <typename CPUJacobianPointTy>
  static gpuError_t Execute(const kernels::msm::ExecutionConfig<Curve>& config,
                            CPUJacobianPointTy* results,
                            CPUJacobianPointTy* out) {
    gpuError_t error = ExecuteAsync(config);
    if (error != gpuSuccess) return error;
    error = gpuStreamSynchronize(config.stream);
    if (error != gpuSuccess) return error;

    gpuMemcpy(results, config.results,
              sizeof(JacobianPoint<Curve>) * kModulusBits, gpuMemcpyDefault);
    *out = Accumulate(results);
    return gpuSuccess;
  }

  template <typename CPUJacobianPointTy>
  static CPUJacobianPointTy Accumulate(const CPUJacobianPointTy* results) {
    CPUJacobianPointTy ret = CPUJacobianPointTy::Zero();
    for (size_t i = 0; i < kModulusBits; ++i) {
      size_t index = kModulusBits - i - 1;
      CPUJacobianPointTy bucket = results[index];
      if (i == 0) {
        ret = bucket;
      } else {
        ret.DoubleInPlace();
        ret += bucket;
      }
    }
    return ret;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_CUDA_H_
