#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_

#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon::math {

template <typename PointTy>
struct MSMTestSet {
  using ScalarField = typename PointTy::ScalarField;
  using ReturnTy =
      typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;

  std::vector<PointTy> bases;
  std::vector<ScalarField> scalars;
  ReturnTy answer;

  static MSMTestSet Random(size_t size, bool use_msm) {
    MSMTestSet test_set;
    test_set.bases =
        base::CreateVector(size, []() { return PointTy::Random(); });
    test_set.scalars =
        base::CreateVector(size, []() { return ScalarField::Random(); });
    test_set.ComputeAnswer(use_msm);
    return test_set;
  }

 private:
  void ComputeAnswer(bool use_msm) {
    answer = ReturnTy::Zero();
    if (use_msm) {
      VariableBaseMSM<PointTy> msm;
      msm.Run(bases, scalars, &answer);
    } else {
      for (size_t i = 0; i < bases.size(); ++i) {
        answer += bases[i].ScalarMul(scalars[i].ToBigInt());
      }
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_
