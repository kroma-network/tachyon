#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_VARIABLE_BASE_MSM_TEST_SET_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_VARIABLE_BASE_MSM_TEST_SET_H_

#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/test/random.h"

namespace tachyon::math {

enum class VariableBaseMSMMethod {
  kNone,
  kMSM,
  kNaive,
};

template <typename Point,
          typename Bucket = typename VariableBaseMSM<Point>::Bucket>
struct VariableBaseMSMTestSet {
  using ScalarField = typename Point::ScalarField;

  std::vector<Point> bases;
  std::vector<ScalarField> scalars;
  Bucket answer;

  constexpr size_t size() const { return bases.size(); }

  static VariableBaseMSMTestSet Random(size_t size,
                                       VariableBaseMSMMethod method) {
    VariableBaseMSMTestSet test_set;
    test_set.bases = CreatePseudoRandomPoints<Point>(size);
    test_set.scalars =
        base::CreateVector(size, []() { return ScalarField::Random(); });
    test_set.ComputeAnswer(method);
    return test_set;
  }

  static VariableBaseMSMTestSet NonUniform(size_t size, size_t scalar_size,
                                           VariableBaseMSMMethod method) {
    VariableBaseMSMTestSet test_set;
    test_set.bases = CreatePseudoRandomPoints<Point>(size);
    std::vector<ScalarField> scalar_sets =
        base::CreateVector(scalar_size, []() { return ScalarField::Random(); });
    test_set.scalars = base::CreateVector(
        size, [&scalar_sets]() { return base::UniformElement(scalar_sets); });
    test_set.ComputeAnswer(method);
    return test_set;
  }

  static VariableBaseMSMTestSet Easy(size_t size,
                                     VariableBaseMSMMethod method) {
    VariableBaseMSMTestSet test_set;
    test_set.bases =
        base::CreateVector(size, []() { return Point::Generator(); });
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
        ss << bases[i].ToString() << std::endl;
      }
      if (!base::WriteFile(dir.Append("bases.txt"), ss.str())) return false;
    }
    std::stringstream ss;
    for (size_t i = 0; i < scalars.size(); ++i) {
      ss << scalars[i].ToString() << std::endl;
    }
    return base::WriteFile(dir.Append("scalars.txt"), ss.str());
  }

 private:
  void ComputeAnswer(VariableBaseMSMMethod method) {
    answer = Bucket::Zero();
    switch (method) {
      case VariableBaseMSMMethod::kNone:
        break;
      case VariableBaseMSMMethod::kMSM: {
        VariableBaseMSM<Point> msm;
        CHECK(msm.Run(bases, scalars, &answer));
        break;
      }
      case VariableBaseMSMMethod::kNaive: {
        using AddResult =
            typename internal::AdditiveSemigroupTraits<Point>::ReturnTy;
        AddResult sum = AddResult::Zero();
        for (size_t i = 0; i < bases.size(); ++i) {
          sum += bases[i] * scalars[i];
        }
        answer = ConvertPoint<Bucket>(sum);
        break;
      }
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_VARIABLE_BASE_MSM_TEST_SET_H_
