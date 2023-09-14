#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_

#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/test/random.h"

namespace tachyon::math {

enum class MSMMethod {
  kNone,
  kMSM,
  kNaive,
};

template <typename PointTy,
          typename Bucket = typename VariableBaseMSM<PointTy>::Bucket>
struct MSMTestSet {
  using ScalarField = typename PointTy::ScalarField;

  std::vector<PointTy> bases;
  std::vector<ScalarField> scalars;
  Bucket answer;

  constexpr size_t size() const { return bases.size(); }

  static MSMTestSet Random(size_t size, MSMMethod method) {
    MSMTestSet test_set;
    test_set.bases = CreatePseudoRandomPoints<PointTy>(size);
    test_set.scalars =
        base::CreateVector(size, []() { return ScalarField::Random(); });
    test_set.ComputeAnswer(method);
    return test_set;
  }

  static MSMTestSet NonUniform(size_t size, size_t scalar_size,
                               MSMMethod method) {
    MSMTestSet test_set;
    test_set.bases = CreatePseudoRandomPoints<PointTy>(size);
    std::vector<ScalarField> scalar_sets =
        base::CreateVector(scalar_size, []() { return ScalarField::Random(); });
    test_set.scalars = base::CreateVector(
        size, [&scalar_sets]() { return base::Uniform(scalar_sets); });
    test_set.ComputeAnswer(method);
    return test_set;
  }

  static MSMTestSet Easy(size_t size, MSMMethod method) {
    MSMTestSet test_set;
    test_set.bases =
        base::CreateVector(size, []() { return PointTy::Generator(); });
    ScalarField s = ScalarField::One();
    test_set.scalars = base::CreateVector(size, [&s]() {
      ScalarField ret = s;
      s += ScalarField::One();
      return ret;
    });
    test_set.ComputeAnswer(method);
    return test_set;
  }

  bool WriteToFile(const base::FilePath& dir) const {
    {
      std::stringstream ss;
      for (size_t i = 0; i < bases.size(); ++i) {
        ss << bases[i] << std::endl;
      }
      if (!base::WriteFile(dir.Append("bases.txt"), ss.str())) return false;
    }
    std::stringstream ss;
    for (size_t i = 0; i < scalars.size(); ++i) {
      ss << scalars[i] << std::endl;
    }
    return base::WriteFile(dir.Append("scalars.txt"), ss.str());
  }

 private:
  void ComputeAnswer(MSMMethod method) {
    answer = Bucket::Zero();
    switch (method) {
      case MSMMethod::kNone:
        break;
      case MSMMethod::kMSM: {
        VariableBaseMSM<PointTy> msm;
        msm.Run(bases, scalars, &answer);
        break;
      }
      case MSMMethod::kNaive: {
        using AddResultTy =
            typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;
        AddResultTy sum = AddResultTy::Zero();
        for (size_t i = 0; i < bases.size(); ++i) {
          sum += bases[i].ScalarMul(scalars[i].ToBigInt());
        }
        answer = ConvertPoint<Bucket>(sum);
        break;
      }
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_MSM_TEST_SET_H_
