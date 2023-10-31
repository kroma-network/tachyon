// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_DENSE_EVALUATIONS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_DENSE_EVALUATIONS_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/multivariate/support_poly_operators.h"

namespace tachyon::math {

template <typename F, size_t N>
class MultilinearDenseEvaluations {
 public:
  constexpr static const size_t kMaxSize = N;

  using Field = F;

  constexpr MultilinearDenseEvaluations()
      : MultilinearDenseEvaluations({F::Zero()}) {}
  constexpr explicit MultilinearDenseEvaluations(
      const std::vector<F>& evaluations)
      : evaluations_(evaluations) {
    CHECK_LE(evaluations_.size(), N);
  }
  constexpr explicit MultilinearDenseEvaluations(std::vector<F>&& evaluations)
      : evaluations_(std::move(evaluations)) {
    CHECK_LE(evaluations_.size(), N);
  }

  constexpr static MultilinearDenseEvaluations Zero(size_t size) {
    return MultilinearDenseEvaluations(base::CreateVector(size, F::Zero()));
  }

  constexpr static MultilinearDenseEvaluations One(size_t size) {
    return MultilinearDenseEvaluations(base::CreateVector(size, F::One()));
  }

  constexpr static MultilinearDenseEvaluations Random(size_t size) {
    return MultilinearDenseEvaluations(
        base::CreateVector(size, []() { return F::Random(); }));
  }

  constexpr bool operator==(const MultilinearDenseEvaluations& other) const {
    return evaluations_ == other.evaluations_;
  }

  constexpr bool operator!=(const MultilinearDenseEvaluations& other) const {
    return !operator==(other);
  }

  constexpr F* Get(size_t i) {
    return const_cast<F*>(std::as_const(*this).Get(i));
  }

  constexpr const F* Get(size_t i) const {
    if (i < evaluations_.size()) {
      return &evaluations_[i];
    }
    return nullptr;
  }

  constexpr bool IsZero() const {
    return std::all_of(evaluations_.begin(), evaluations_.end(),
                       [](const F& value) { return value.IsZero(); });
  }

  constexpr bool IsOne() const {
    return std::all_of(evaluations_.begin(), evaluations_.end(),
                       [](const F& value) { return value.IsOne(); });
  }

  constexpr size_t Degree() const {
    return base::bits::SafeLog2Ceiling(evaluations_.size());
  }

  // Fix k variables out of n variables, where k is
  // |partial_point.size()| and n is |Degree()|.
  MultilinearDenseEvaluations FixVariables(
      const std::vector<F>& partial_point) const {
    size_t k = partial_point.size();
    size_t n = Degree();
    CHECK_LE(k, n);
    std::vector<F> poly = evaluations_;

    // clang-format off
    // P(x₀, x₁) = 1(1 - x₀)(1 - x₁) + 2x₀(1 - x₁) + 3(1 - x₀)x₁ + 4x₀x₁
    //
    // Fixing s₀:
    // P(s₀, x₁) = 1(1 - s₀)(1 - x₁) + 2s₀(1 - x₁) + 3(1 - s₀)x₁ + 4s₀x₁
    //           = (1(1 - s₀) + 2s₀)(1 - x₁) + (3(1 - s₀) + 4s₀)x₁
    //           = (left₀(1 - s₀) + right₀s₀)(1 - x₁) + (left₁(1 - s₀) + right₁s₀)x₁
    //             (where left₀ = 1, right₀ = 2, left₁ = 3 and right₁ = 4)
    //           = (right₀ + s₀(right₀ - left₀))(1 - x₁) + (right₁ + s₀(right₁ - left₁))x₁
    //
    // Fixing s₁:
    // P(s₀, s₁) = (1 + (2 - 1)s₀)(1 - s₁) + (3 + (4 - 1)4s₀)s₁
    //           = (left₀(1 - s₀) + right₀s₀)(1 - x₁) + (left₁(1 - s₀) + right₁s₀)x₁
    //             (where left₀ = 1 + (2 - 1)s₀ and right₀ = (3 + (4 - 1)4s₀)
    //           = right₀ + s₁(right₀ - left₀)
    // clang-format on
    for (size_t i = 1; i <= k; ++i) {
      const F& r = partial_point[i - 1];
      for (size_t b = 0; b < (size_t{1} << (n - i)); ++b) {
        const F& left = poly[b << 1];
        const F& right = poly[(b << 1) + 1];
        poly[b] = left + r * (right - left);
      }
    }
    return MultilinearDenseEvaluations(
        std::vector<F>(poly.begin(), poly.begin() + (size_t{1} << (n - k))));
  }

  // Evaluate polynomial at |point|. It uses |FixVariables()| internally. The
  // |point| is a vector in {0, 1}ᵏ in little-endian form. If the size of
  // |point| is less than the degree of the polynomial, the remaining components
  // of |point| are assumed to be zeros.
  //
  //   MultilinearDenseEvaluations<GF7, 3> evals
  //       MultilinearDenseEvaluations<GF7, 3>::Random();
  //   GF7 a = evals.Evaluate({GF7(2), GF7(3)});
  //   GF7 b = evals.Evaluate({GF7(2), GF7(3), GF7(0)});
  //   CHECK_EQ(a, b);
  F Evaluate(const std::vector<F>& point) const {
    CHECK_EQ(
        point.size(),
        static_cast<size_t>(base::bits::SafeLog2Ceiling(evaluations_.size())));

    MultilinearDenseEvaluations fixed = FixVariables(point);

    if (fixed.IsZero()) return F::Zero();
    return fixed.evaluations_[0];
  }

  std::string ToString() const { return base::VectorToString(evaluations_); }

 private:
  friend class internal::MultilinearExtensionOp<
      MultilinearDenseEvaluations<F, N>>;

  std::vector<F> evaluations_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_DENSE_EVALUATIONS_H_
