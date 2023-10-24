#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_

#include <algorithm>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/math/polynomials/multivariate/multilinear_extension.h"

namespace tachyon::math {
namespace internal {

template <typename F, size_t MaxDegree>
class MultilinearExtensionOp<MultilinearDenseEvaluations<F, MaxDegree>> {
 public:
  using D = MultilinearDenseEvaluations<F, MaxDegree>;

  static MultilinearExtension<D>& AddInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] += r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D>& SubInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] -= r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D>& NegInPlace(MultilinearExtension<D>& self) {
    std::vector<F>& evaluations = self.evaluations_.evaluations_;
    // clang-format off
    OPENMP_PARALLEL_FOR(F& evaluation : evaluations) {
      // clang-format on
      evaluation.NegInPlace();
    }
    return self;
  }

  static MultilinearExtension<D>& MulInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      l_evaluations[i] *= r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D>& DivInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      l_evaluations[i] /= r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D> ToDense(const MultilinearExtension<D>& self) {
    return self;
  }
};

}  // namespace internal
}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_
