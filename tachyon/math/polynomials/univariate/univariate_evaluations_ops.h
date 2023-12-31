// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_OPS_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::math {
namespace internal {

template <typename F, size_t MaxDegree>
class UnivariateEvaluationsOp {
 public:
  using Poly = UnivariateEvaluations<F, MaxDegree>;

  static Poly& AddInPlace(Poly& self, const Poly& other) {
    std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty()) {
      // 0 + g(x)
      l_evaluations = r_evaluations;
      return self;
    }
    // f(x) + 0 skips this for loop.
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] += r_evaluations[i];
    }
    return self;
  }

  static Poly& SubInPlace(Poly& self, const Poly& other) {
    std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty()) {
      // 0 - g(x)
      l_evaluations.resize(r_evaluations.size());
    }
    // f(x) - 0 skips this for loop.
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] -= r_evaluations[i];
    }
    return self;
  }

  static Poly& NegInPlace(Poly& self) {
    std::vector<F>& evaluations = self.evaluations_;
    // clang-format off
    OPENMP_PARALLEL_FOR(F& evaluation : evaluations) {
      // clang-format on
      evaluation.NegInPlace();
    }
    return self;
  }

  static Poly& MulInPlace(Poly& self, const Poly& other) {
    std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty()) {
      // 0 * g(x)
      return self;
    }
    if (r_evaluations.empty()) {
      // f(x) * 0
      l_evaluations.clear();
      return self;
    }
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] *= r_evaluations[i];
    }
    return self;
  }

  static Poly& MulInPlace(Poly& self, const F& scalar) {
    std::vector<F>& l_evaluations = self.evaluations_;
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      l_evaluations[i] *= scalar;
    }
    return self;
  }

  static Poly& DivInPlace(Poly& self, const Poly& other) {
    std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty()) {
      // 0 / g(x)
      return self;
    }
    // TODO(chokobole): Check division by zero polynomial.
    // f(x) / 0 skips this for loop.
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] /= r_evaluations[i];
    }
    return self;
  }

  static Poly& DivInPlace(Poly& self, const F& scalar) {
    std::vector<F>& l_evaluations = self.evaluations_;
    F scalar_inv = scalar.Inverse();
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      l_evaluations[i] *= scalar_inv;
    }
    return self;
  }
};

}  // namespace internal
}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_OPS_H_
