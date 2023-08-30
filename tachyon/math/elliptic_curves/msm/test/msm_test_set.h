#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_

#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon::math {

enum class MSMMethod {
  kNone,
  kMSM,
  kNaive,
};

template <typename PointTy>
struct MSMTestSet {
  using ScalarField = typename PointTy::ScalarField;
  using ReturnTy =
      typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;

  std::vector<PointTy> bases;
  std::vector<ScalarField> scalars;
  ReturnTy answer;

  static MSMTestSet Random(size_t size, MSMMethod method) {
    MSMTestSet test_set;
    test_set.bases =
        base::CreateVector(size, []() { return PointTy::Random(); });
    test_set.scalars =
        base::CreateVector(size, []() { return ScalarField::Random(); });
    test_set.ComputeAnswer(method);
    return test_set;
  }

  static MSMTestSet NonUniform(size_t size, size_t scalar_size,
                               MSMMethod method) {
    MSMTestSet test_set;
    test_set.bases =
        base::CreateVector(size, []() { return PointTy::Random(); });
    std::vector<ScalarField> scalar_sets =
        base::CreateVector(scalar_size, []() { return ScalarField::Random(); });
    test_set.scalars = base::CreateVector(
        size, [&scalar_sets]() { return base::Uniform(scalar_sets); });
    test_set.ComputeAnswer(method);
    return test_set;
  }

 private:
  void ComputeAnswer(MSMMethod method) {
    answer = ReturnTy::Zero();
    switch (method) {
      case MSMMethod::kNone:
        break;
      case MSMMethod::kMSM: {
        VariableBaseMSM<PointTy> msm;
        msm.Run(bases, scalars, &answer);
        break;
      }
      case MSMMethod::kNaive: {
        for (size_t i = 0; i < bases.size(); ++i) {
          answer += bases[i].ScalarMul(scalars[i].ToBigInt());
        }
        break;
      }
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_
