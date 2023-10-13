#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_

#include <utility>

#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_adapter.h"

namespace tachyon::math {
template <typename PointTy>
class VariableBaseMSM {
 public:
  using ScalarField = typename PointTy::ScalarField;
  using Bucket = typename Pippenger<PointTy>::Bucket;

  // MSM(Multi-Scalar Multiplication): s₀ * g₀ + s₁ * g₁ + ... + sₙ * gₙ
  // Variable-base MSM is an operation that multiplies different base points
  // with respective scalars, unlike the Fixed-base MSM, which uses the same
  // base point for all multiplications.
  // This implementation uses Pippenger's algorithm to compute the MSM.
  template <typename BaseInputIterator, typename ScalarInputIterator>
  bool Run(BaseInputIterator bases_first, BaseInputIterator bases_last,
           ScalarInputIterator scalars_first, ScalarInputIterator scalars_last,
           Bucket* ret) {
    PippengerAdapter<PointTy> pippenger;
    return pippenger.Run(std::move(bases_first), std::move(bases_last),
                         std::move(scalars_first), std::move(scalars_last),
                         ret);
  }

  template <typename BaseContainer, typename ScalarContainer>
  bool Run(const BaseContainer& bases, const ScalarContainer& scalars,
           Bucket* ret) {
    return Run(std::begin(bases), std::end(bases), std::begin(scalars),
               std::end(scalars), ret);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_
