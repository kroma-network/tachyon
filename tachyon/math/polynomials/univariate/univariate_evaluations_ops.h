// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_OPS_H_

#include <algorithm>
#include <atomic>
#include <optional>
#include <utility>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::math {
namespace internal {

template <typename F, size_t MaxDegree>
class UnivariateEvaluationsOp {
 public:
  using Poly = UnivariateEvaluations<F, MaxDegree>;

  static Poly Add(const Poly& self, const Poly& other) {
    const std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty()) {
      // 0 + g(x)
      return other;
    }
    if (r_evaluations.empty()) {
      // f(x) + 0
      return self;
    }
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    std::vector<F> o_evaluations(r_evaluations.size());
    OMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      o_evaluations[i] = l_evaluations[i] + r_evaluations[i];
    }
    return Poly(std::move(o_evaluations));
  }

  static Poly& AddInPlace(Poly& self, const Poly& other) {
    std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty()) {
      // 0 + g(x)
      return self = other;
    }
    if (r_evaluations.empty()) {
      // f(x) + 0
      return self;
    }
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] += r_evaluations[i];
    }
    return self;
  }

  static Poly Sub(const Poly& self, const Poly& other) {
    const std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty()) {
      // 0 - g(x)
      return -other;
    }
    if (r_evaluations.empty()) {
      // f(x) - 0
      return self;
    }
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    std::vector<F> o_evaluations(r_evaluations.size());
    OMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      o_evaluations[i] = l_evaluations[i] - r_evaluations[i];
    }
    return Poly(std::move(o_evaluations));
  }

  static Poly& SubInPlace(Poly& self, const Poly& other) {
    std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty()) {
      // 0 - g(x)
      return self = -other;
    }
    if (r_evaluations.empty()) {
      // f(x) - 0
      return self;
    }
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] -= r_evaluations[i];
    }
    return self;
  }

  static Poly Negate(const Poly& self) {
    const std::vector<F>& i_evaluations = self.evaluations_;
    if (i_evaluations.empty()) {
      return self;
    }
    std::vector<F> o_evaluations(i_evaluations.size());
    OMP_PARALLEL_FOR(size_t i = 0; i < i_evaluations.size(); ++i) {
      o_evaluations[i] = -i_evaluations[i];
    }
    return Poly(std::move(o_evaluations));
  }

  static Poly& NegateInPlace(Poly& self) {
    std::vector<F>& evaluations = self.evaluations_;
    if (evaluations.empty()) {
      return self;
    }
    // clang-format off
    OMP_PARALLEL_FOR(F& evaluation : evaluations) {
      // clang-format on
      evaluation.NegateInPlace();
    }
    return self;
  }

  static Poly Mul(const Poly& self, const Poly& other) {
    const std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    if (l_evaluations.empty() || r_evaluations.empty()) {
      // 0 * g(x) or f(x) * 0
      return Poly::Zero();
    }
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    std::vector<F> o_evaluations(r_evaluations.size());
    OMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      o_evaluations[i] = l_evaluations[i] * r_evaluations[i];
    }
    return Poly(std::move(o_evaluations));
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
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] *= r_evaluations[i];
    }
    return self;
  }

  static Poly Mul(const Poly& self, const F& scalar) {
    const std::vector<F>& l_evaluations = self.evaluations_;
    if (l_evaluations.empty() || scalar.IsZero()) {
      // 0 * s or f(x) * 0
      return Poly::Zero();
    }
    if (scalar.IsOne()) {
      // f(x) * 1
      return self;
    }
    std::vector<F> o_evaluations(l_evaluations.size());
    OMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      o_evaluations[i] = l_evaluations[i] * scalar;
    }
    return Poly(std::move(o_evaluations));
  }

  static Poly& MulInPlace(Poly& self, const F& scalar) {
    std::vector<F>& l_evaluations = self.evaluations_;
    if (l_evaluations.empty() || scalar.IsOne()) {
      // 0 * s or f(x) * 1
      return self;
    }
    OMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      l_evaluations[i] *= scalar;
    }
    return self;
  }

  CONSTEXPR_IF_NOT_OPENMP static std::optional<Poly> Div(const Poly& self,
                                                         const Poly& other) {
    const std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    // f(x) / 0
    if (UNLIKELY(r_evaluations.empty())) {
      LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
      return std::nullopt;
    }
    // 0 / g(x)
    if (l_evaluations.empty()) {
      return self;
    }
    // f(x) & g(x) unequal evaluation sizes
    if (UNLIKELY(l_evaluations.size() != r_evaluations.size())) {
      LOG_IF_NOT_GPU(ERROR) << "Evaluation sizes unequal for division";
      return std::nullopt;
    }
    std::vector<F> o_evaluations(r_evaluations.size());
    std::atomic<bool> check_valid(true);
    OMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      const std::optional<F> div = l_evaluations[i] / r_evaluations[i];
      if (UNLIKELY(!div)) {
        check_valid.store(false, std::memory_order_relaxed);
        continue;
      }
      o_evaluations[i] = std::move(*div);
    }
    if (LIKELY(check_valid.load(std::memory_order_relaxed))) {
      return Poly(std::move(o_evaluations));
    }
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  [[nodiscard]] CONSTEXPR_IF_NOT_OPENMP static std::optional<Poly*> DivInPlace(
      Poly& self, const Poly& other) {
    std::vector<F>& l_evaluations = self.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_;
    // f(x) / 0
    if (UNLIKELY(r_evaluations.empty())) {
      LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
      return std::nullopt;
    }
    // 0 / g(x)
    if (l_evaluations.empty()) {
      return &self;
    }
    // f(x) & g(x) unequal evaluation sizes
    if (UNLIKELY(l_evaluations.size() != r_evaluations.size())) {
      LOG_IF_NOT_GPU(ERROR) << "Evaluation sizes unequal for division";
      return std::nullopt;
    }
    std::atomic<bool> check_valid(true);
    OMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      if (UNLIKELY(!(l_evaluations[i] /= r_evaluations[i])))
        check_valid.store(false, std::memory_order_relaxed);
    }
    if (LIKELY(check_valid.load(std::memory_order_relaxed))) {
      return &self;
    }
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  constexpr static std::optional<Poly> Div(const Poly& self, const F& scalar) {
    const std::optional<F> scalar_inv = scalar.Inverse();
    if (LIKELY(scalar_inv)) return Mul(self, *scalar_inv);
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  [[nodiscard]] constexpr static std::optional<Poly*> DivInPlace(
      Poly& self, const F& scalar) {
    const std::optional<F> scalar_inv = scalar.Inverse();
    if (LIKELY(scalar_inv)) return &MulInPlace(self, *scalar_inv);
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }
};

}  // namespace internal
}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_OPS_H_
