#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_

#include <algorithm>
#include <atomic>
#include <optional>
#include <utility>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/math/base/invalid_operation.h"
#include "tachyon/math/polynomials/multivariate/multilinear_extension.h"

namespace tachyon::math {
namespace internal {

template <typename F, size_t MaxDegree>
class MultilinearExtensionOp<MultilinearDenseEvaluations<F, MaxDegree>> {
 public:
  using D = MultilinearDenseEvaluations<F, MaxDegree>;

  static MultilinearExtension<D> Add(const MultilinearExtension<D>& self,
                                     const MultilinearExtension<D>& other) {
    const std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
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
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      o_evaluations[i] = l_evaluations[i] + r_evaluations[i];
    }
    return MultilinearExtension<D>(D(std::move(o_evaluations)));
  }

  static MultilinearExtension<D>& AddInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    if (l_evaluations.empty()) {
      // 0 + g(x)
      return self = other;
    }
    if (r_evaluations.empty()) {
      // f(x) + 0
      return self;
    }
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] += r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D> Sub(const MultilinearExtension<D>& self,
                                     const MultilinearExtension<D>& other) {
    const std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
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
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      o_evaluations[i] = l_evaluations[i] - r_evaluations[i];
    }
    return MultilinearExtension<D>(D(std::move(o_evaluations)));
  }

  static MultilinearExtension<D>& SubInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    if (l_evaluations.empty()) {
      // 0 - g(x)
      return self = -other;
    }
    if (r_evaluations.empty()) {
      // f(x) - 0
      return self;
    }
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      l_evaluations[i] -= r_evaluations[i];
    }
    return self;
  }

  static MultilinearExtension<D> Negate(const MultilinearExtension<D>& self) {
    const std::vector<F>& i_evaluations = self.evaluations_.evaluations_;
    if (i_evaluations.empty()) {
      return self;
    }
    std::vector<F> o_evaluations(i_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < o_evaluations.size(); ++i) {
      o_evaluations[i] = -i_evaluations[i];
    }
    return MultilinearExtension<D>(D(std::move(o_evaluations)));
  }

  static MultilinearExtension<D>& NegateInPlace(MultilinearExtension<D>& self) {
    std::vector<F>& evaluations = self.evaluations_.evaluations_;
    if (evaluations.empty()) {
      return self;
    }
    // clang-format off
    OPENMP_PARALLEL_FOR(F& evaluation : evaluations) {
      // clang-format on
      evaluation.NegateInPlace();
    }
    return self;
  }

  static MultilinearExtension<D> Mul(const MultilinearExtension<D>& self,
                                     const MultilinearExtension<D>& other) {
    const std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    if (l_evaluations.empty() || r_evaluations.empty()) {
      // 0 * g(x) or f(x) * 0
      return MultilinearExtension<D>::Zero();
    }
    CHECK_EQ(l_evaluations.size(), r_evaluations.size());
    std::vector<F> o_evaluations(r_evaluations.size());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      o_evaluations[i] = l_evaluations[i] * r_evaluations[i];
    }
    return MultilinearExtension<D>(D(std::move(o_evaluations)));
  }

  static MultilinearExtension<D>& MulInPlace(
      MultilinearExtension<D>& self, const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
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
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      l_evaluations[i] *= r_evaluations[i];
    }
    return self;
  }

  CONSTEXPR_IF_NOT_OPENMP static std::optional<MultilinearExtension<D>> Div(
      const MultilinearExtension<D>& self,
      const MultilinearExtension<D>& other) {
    const std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    // f(x) / 0
    if (UNLIKELY(InvalidOperation(r_evaluations.empty(),
                                  "Division by zero attempted"))) {
      return std::nullopt;
    }
    // 0 / g(x)
    if (l_evaluations.empty()) {
      return self;
    }
    // f(x) & g(x) unequal evaluation sizes
    if (UNLIKELY(InvalidOperation(l_evaluations.size() != r_evaluations.size(),
                                  "Evaluation sizes unequal for division"))) {
      return std::nullopt;
    }
    std::vector<F> o_evaluations(r_evaluations.size());
    std::atomic<bool> check_valid(true);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < l_evaluations.size(); ++i) {
      std::optional<F> div = l_evaluations[i] / r_evaluations[i];
      if (UNLIKELY(!div)) {
        check_valid.store(false, std::memory_order_relaxed);
        continue;
      }
      o_evaluations[i] = std::move(div).value();
    }
    if (UNLIKELY(InvalidOperation(!check_valid.load(std::memory_order_relaxed),
                                  "Division by zero attempted"))) {
      return std::nullopt;
    }
    return MultilinearExtension<D>(D(std::move(o_evaluations)));
  }

  [[nodiscard]] CONSTEXPR_IF_NOT_OPENMP static std::optional<
      MultilinearExtension<D>*>
  DivInPlace(MultilinearExtension<D>& self,
             const MultilinearExtension<D>& other) {
    std::vector<F>& l_evaluations = self.evaluations_.evaluations_;
    const std::vector<F>& r_evaluations = other.evaluations_.evaluations_;
    // f(x) / 0
    if (UNLIKELY(InvalidOperation(r_evaluations.empty(),
                                  "Division by zero attempted"))) {
      return std::nullopt;
    }
    // 0 / g(x)
    if (l_evaluations.empty()) {
      return &self;
    }
    // f(x) & g(x) unequal evaluation sizes
    if (UNLIKELY(InvalidOperation(l_evaluations.size() != r_evaluations.size(),
                                  "Evaluation sizes unequal for division"))) {
      return std::nullopt;
    }
    std::atomic<bool> check_valid(true);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < r_evaluations.size(); ++i) {
      if (UNLIKELY(!(l_evaluations[i] /= r_evaluations[i])))
        check_valid.store(false, std::memory_order_relaxed);
    }
    if (UNLIKELY(InvalidOperation(!check_valid.load(std::memory_order_relaxed),
                                  "Division by zero attempted"))) {
      return std::nullopt;
    }
    return &self;
  }

  static const MultilinearExtension<D>& ToDense(
      const MultilinearExtension<D>& self) {
    return self;
  }
};

}  // namespace internal
}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_OPS_H_
