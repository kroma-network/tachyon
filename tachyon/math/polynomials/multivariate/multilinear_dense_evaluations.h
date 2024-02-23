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

#include "absl/hash/hash.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/multivariate/support_poly_operators.h"

namespace tachyon {
namespace math {

template <typename F, size_t MaxDegree>
class MultilinearDenseEvaluations {
 public:
  constexpr static size_t kMaxDegree = MaxDegree;
  constexpr static F kZero = F::Zero();

  using Field = F;
  using Point = std::vector<F>;

  constexpr MultilinearDenseEvaluations() = default;
  constexpr explicit MultilinearDenseEvaluations(
      const std::vector<F>& evaluations)
      : evaluations_(evaluations) {
    CHECK_LE(Degree(), MaxDegree);
  }
  constexpr explicit MultilinearDenseEvaluations(std::vector<F>&& evaluations)
      : evaluations_(std::move(evaluations)) {
    CHECK_LE(Degree(), MaxDegree);
  }

  // NOTE(chokobole): The zero polynomial can be represented in two forms:
  // 1. An empty vector.
  // 2. A vector filled with |F::Zero()|.
  constexpr static MultilinearDenseEvaluations Zero() {
    return MultilinearDenseEvaluations();
  }

  constexpr static MultilinearDenseEvaluations One(size_t degree) {
    return MultilinearDenseEvaluations(
        base::CreateVector(size_t{1} << degree, F::One()));
  }

  constexpr static MultilinearDenseEvaluations Random(size_t degree) {
    return MultilinearDenseEvaluations(
        base::CreateVector(size_t{1} << degree, []() { return F::Random(); }));
  }

  constexpr const std::vector<F>& evaluations() const { return evaluations_; }
  constexpr std::vector<F>& evaluations() { return evaluations_; }

  constexpr bool operator==(const MultilinearDenseEvaluations& other) const {
    if (evaluations_.empty()) {
      return other.IsZero();
    }
    if (other.evaluations_.empty()) {
      return IsZero();
    }
    return evaluations_ == other.evaluations_;
  }

  constexpr bool operator!=(const MultilinearDenseEvaluations& other) const {
    return !operator==(other);
  }

  constexpr F& at(size_t i) {
    CHECK_LT(i, evaluations_.size());
    return evaluations_[i];
  }

  constexpr const F& at(size_t i) const { return (*this)[i]; }

  constexpr const F& operator[](size_t i) const {
    if (i < evaluations_.size()) {
      return evaluations_[i];
    }
    return kZero;
  }

  constexpr bool IsZero() const {
    if (evaluations_.empty()) return true;
    return std::all_of(evaluations_.begin(), evaluations_.end(),
                       [](const F& value) { return value.IsZero(); });
  }

  constexpr bool IsOne() const {
    if (evaluations_.empty()) return false;
    return std::all_of(evaluations_.begin(), evaluations_.end(),
                       [](const F& value) { return value.IsOne(); });
  }

  constexpr size_t Degree() const {
    return base::bits::SafeLog2Ceiling(evaluations_.size());
  }

  // Fix k variables out of n variables, where k is
  // |partial_point.size()| and n is |Degree()|.
  MultilinearDenseEvaluations FixVariables(const Point& partial_point) const {
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
  F Evaluate(const Point& point) const {
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
      MultilinearDenseEvaluations<F, MaxDegree>>;

  // NOTE(chokobole): This creates a polynomial that contains |F::Zero()| up to
  // |degree| + 1.
  constexpr static MultilinearDenseEvaluations Zero(size_t degree) {
    MultilinearDenseEvaluations ret;
    ret.evaluations_ = base::CreateVector(size_t{1} << degree, F::Zero());
    return ret;
  }

  std::vector<F> evaluations_;
};

template <typename H, typename F, size_t MaxDegree>
H AbslHashValue(H h, const MultilinearDenseEvaluations<F, MaxDegree>& evals) {
  // NOTE(chokobole): We shouldn't hash only with a non-zero term.
  // See https://abseil.io/docs/cpp/guides/hash#the-abslhashvalue-overload
  size_t degree = 0;
  for (const F& eval : evals.evaluations()) {
    h = H::combine(std::move(h), eval);
    ++degree;
  }
  F zero = F::Zero();
  for (size_t i = degree; i < size_t{1} << MaxDegree; ++i) {
    h = H::combine(std::move(h), zero);
  }
  return h;
}

}  // namespace math

namespace base {

template <typename F, size_t MaxDegree>
class Copyable<math::MultilinearDenseEvaluations<F, MaxDegree>> {
 public:
  static bool WriteTo(
      const math::MultilinearDenseEvaluations<F, MaxDegree>& evals,
      Buffer* buffer) {
    return buffer->Write(evals.evaluations());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       math::MultilinearDenseEvaluations<F, MaxDegree>* evals) {
    std::vector<F> evaluations;
    if (!buffer.Read(&evaluations)) return false;
    *evals =
        math::MultilinearDenseEvaluations<F, MaxDegree>(std::move(evaluations));
    return true;
  }

  static size_t EstimateSize(
      const math::MultilinearDenseEvaluations<F, MaxDegree>& evals) {
    return base::EstimateSize(evals.evaluations());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_DENSE_EVALUATIONS_H_
