#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_MSM_ALGORITHM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_MSM_ALGORITHM_H_

#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/elliptic_curves/jacobian_point.h"

namespace tachyon::math {

enum class MSMAlgorithmKind {
  kBellmanMSM,
  kCUZK,
  kIcicle,
  kPippenger,
};

template <typename GpuCurve>
class MSMGpuAlgorithm {
 public:
  using ScalarField = typename JacobianPoint<GpuCurve>::ScalarField;
  using CpuCurve = typename GpuCurve::CpuCurve;

  virtual bool Run(const device::gpu::GpuMemory<AffinePoint<GpuCurve>>& bases,
                   const device::gpu::GpuMemory<ScalarField>& scalars,
                   size_t size, JacobianPoint<CpuCurve>* cpu_result) = 0;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_MSM_ALGORITHM_H_
