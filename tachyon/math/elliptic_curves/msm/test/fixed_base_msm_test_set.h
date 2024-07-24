#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_FIXED_BASE_MSM_TEST_SET_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_FIXED_BASE_MSM_TEST_SET_H_

#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/elliptic_curves/msm/fixed_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/test/random.h"

namespace tachyon::math {

enum class FixedBaseMSMMethod {
  kNone,
  kMSM,
  kNaive,
};

template <typename Point>
struct FixedBaseMSMTestSet {
  using AddResult = typename internal::AdditiveSemigroupTraits<Point>::ReturnTy;
  using ScalarField = typename Point::ScalarField;

  Point base;
  std::vector<ScalarField> scalars;
  std::vector<AddResult> answer;

  constexpr size_t size() const { return scalars.size(); }

  static FixedBaseMSMTestSet Random(size_t size, FixedBaseMSMMethod method) {
    FixedBaseMSMTestSet test_set;
    test_set.base = Point::Random();
    test_set.scalars = base::CreateVectorParallel(
        size, []() { return ScalarField::Random(); });
    test_set.ComputeAnswer(method);
    return test_set;
  }

  static FixedBaseMSMTestSet Easy(size_t size, FixedBaseMSMMethod method) {
    FixedBaseMSMTestSet test_set;
    test_set.base = Point::Random();
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
      ss << base.ToString() << std::endl;
      if (!base::WriteFile(dir.Append("base.txt"), ss.str())) return false;
    }
    std::stringstream ss;
    for (size_t i = 0; i < scalars.size(); ++i) {
      ss << scalars[i].ToString() << std::endl;
    }
    return base::WriteFile(dir.Append("scalars.txt"), ss.str());
  }

 private:
  void ComputeAnswer(FixedBaseMSMMethod method) {
    switch (method) {
      case FixedBaseMSMMethod::kNone:
        break;
      case FixedBaseMSMMethod::kMSM: {
        FixedBaseMSM<Point> msm;
        msm.Reset(scalars.size(), base);
        answer.resize(scalars.size());
        CHECK(msm.Run(scalars, &answer));
        break;
      }
      case FixedBaseMSMMethod::kNaive: {
        for (size_t i = 0; i < scalars.size(); ++i) {
          answer.push_back(base * scalars[i]);
        }
        break;
      }
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_TEST_FIXED_BASE_MSM_TEST_SET_H_
