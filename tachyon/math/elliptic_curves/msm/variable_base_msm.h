#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_

#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_adapter.h"

namespace tachyon::math {
template <typename PointTy>
class VariableBaseMSM {
 public:
  using ScalarField = typename PointTy::ScalarField;
  using Bucket = typename Pippenger<PointTy>::Bucket;

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
  bool Run(BaseContainer&& bases, ScalarContainer&& scalars, Bucket* ret) {
    return Run(std::begin(std::forward<BaseContainer>(bases)),
               std::end(std::forward<BaseContainer>(bases)),
               std::begin(std::forward<ScalarContainer>(scalars)),
               std::end(std::forward<ScalarContainer>(scalars)), ret);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_VARIABLE_BASE_MSM_H_
