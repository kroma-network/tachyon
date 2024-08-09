#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_

#include <memory>

#include "tachyon/math/elliptic_curves/msm/algorithms/icicle/icicle_msm.h"

namespace tachyon::math {

template <typename Point>
class VariableBaseMSMGpu {
 public:
  using Curve = typename Point::Curve;
  using Bucket = ProjectivePoint<Curve>;

  VariableBaseMSMGpu(gpuMemPool_t mem_pool, gpuStream_t stream)
      : impl_(std::make_unique<IcicleMSM<Point>>(mem_pool, stream)) {}
  VariableBaseMSMGpu(const VariableBaseMSMGpu& other) = delete;
  VariableBaseMSMGpu& operator=(const VariableBaseMSMGpu& other) = delete;

  template <typename BaseContainer, typename ScalarContainer>
  [[nodiscard]] bool Run(const BaseContainer& bases,
                         const ScalarContainer& cpu_scalars,
                         ProjectivePoint<Curve>* cpu_result) {
    return impl_->Run(bases, cpu_scalars, cpu_result);
  }

 private:
  std::unique_ptr<IcicleMSM<Point>> impl_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_GPU_H_
