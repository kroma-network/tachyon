// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/polynomials/polynomial.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_forwards.h"

namespace tachyon {
namespace math {
namespace internal {

template <typename F, size_t MaxDegree>
class UnivariateEvaluationsOp;

}  // namespace internal

// UnivariateEvaluations represents a univariate polynomial in evaluation form.
// For a univariate polynomial like 3x² + 2x + 1, it can be represented as [(0,
// 1), (1, 6), (2, 17)]. Using Lagrange interpolation, we can easily convert it
// into coefficient form, which is [1, 2, 3]. UnivariateEvaluations only stores
// the y-coordinates, such as [1, 6, 17] in this example. Depending on its
// evaluation domain, the univariate polynomial can vary.
// For more information, refer to
// https://en.wikipedia.org/wiki/Lagrange_polynomial
template <typename F, size_t MaxDegree>
class UnivariateEvaluations final
    : public Polynomial<UnivariateEvaluations<F, MaxDegree>> {
 public:
  constexpr static size_t kMaxDegree = MaxDegree;

  using Field = F;

  constexpr UnivariateEvaluations() : UnivariateEvaluations({F::Zero()}) {}
  constexpr explicit UnivariateEvaluations(const std::vector<F>& evaluations)
      : evaluations_(evaluations) {
    CHECK_LE(Degree(), MaxDegree);
  }
  constexpr explicit UnivariateEvaluations(std::vector<F>&& evaluations)
      : evaluations_(std::move(evaluations)) {
    CHECK_LE(Degree(), MaxDegree);
  }

  constexpr static bool IsCoefficientForm() { return false; }

  constexpr static bool IsEvaluationForm() { return true; }

  constexpr static UnivariateEvaluations Zero(size_t degree) {
    UnivariateEvaluations ret;
    ret.evaluations_ = base::CreateVector(degree + 1, F::Zero());
    return ret;
  }

  // NOTE(chokobole): This is only used at |EvaluationDomain|.
  // See also univariate_polynomial.h.
  constexpr static UnivariateEvaluations UnsafeZero(size_t degree) {
    return Zero(degree);
  }

  constexpr static UnivariateEvaluations One(size_t degree) {
    UnivariateEvaluations ret;
    ret.evaluations_ = base::CreateVector(degree + 1, F::One());
    return ret;
  }

  constexpr static UnivariateEvaluations Random(size_t degree) {
    return UnivariateEvaluations(
        base::CreateVector(degree + 1, []() { return F::Random(); }));
  }

  const std::vector<F>& evaluations() const { return evaluations_; }

  constexpr bool IsZero() const {
    return std::all_of(evaluations_.begin(), evaluations_.end(),
                       [](const F& value) { return value.IsZero(); });
  }

  constexpr bool IsOne() const {
    return std::all_of(evaluations_.begin(), evaluations_.end(),
                       [](const F& value) { return value.IsOne(); });
  }

  constexpr size_t Degree() const { return evaluations_.size() - 1; }

  constexpr bool operator==(const UnivariateEvaluations& other) const {
    return evaluations_ == other.evaluations_;
  }

  constexpr bool operator!=(const UnivariateEvaluations& other) const {
    return !operator==(other);
  }

  constexpr F* operator[](size_t i) {
    return const_cast<F*>(std::as_const(*this)[i]);
  }

  constexpr const F* operator[](size_t i) const {
    if (i < evaluations_.size()) {
      return &evaluations_[i];
    }
    return nullptr;
  }

  std::string ToString() const { return base::VectorToString(evaluations_); }

  // AdditiveSemigroup methods
  UnivariateEvaluations& AddInPlace(const UnivariateEvaluations& other) {
    return internal::UnivariateEvaluationsOp<F, MaxDegree>::AddInPlace(*this,
                                                                       other);
  }

  // AdditiveGroup methods
  UnivariateEvaluations& SubInPlace(const UnivariateEvaluations& other) {
    return internal::UnivariateEvaluationsOp<F, MaxDegree>::SubInPlace(*this,
                                                                       other);
  }

  UnivariateEvaluations& NegInPlace() {
    return internal::UnivariateEvaluationsOp<F, MaxDegree>::NegInPlace(*this);
  }

  // MultiplicativeSemigroup methods
  UnivariateEvaluations& MulInPlace(const UnivariateEvaluations& other) {
    return internal::UnivariateEvaluationsOp<F, MaxDegree>::MulInPlace(*this,
                                                                       other);
  }

  UnivariateEvaluations& DivInPlace(const UnivariateEvaluations& other) {
    return internal::UnivariateEvaluationsOp<F, MaxDegree>::DivInPlace(*this,
                                                                       other);
  }

  constexpr UnivariateEvaluations operator/(
      const UnivariateEvaluations& other) const {
    UnivariateEvaluations poly = *this;
    return poly.DivInPlace(other);
  }

  constexpr UnivariateEvaluations& operator/=(
      const UnivariateEvaluations& other) {
    return DivInPlace(other);
  }

 private:
  friend class internal::UnivariateEvaluationsOp<F, MaxDegree>;
  friend class Radix2EvaluationDomain<F, MaxDegree>;
  friend class MixedRadixEvaluationDomain<F, kMaxDegree>;

  std::vector<F> evaluations_;
};

template <typename F, size_t MaxDegree>
class PolynomialTraits<UnivariateEvaluations<F, MaxDegree>> {
 public:
  constexpr static bool kIsEvaluationForm = true;
};

}  // namespace math

namespace base {

template <typename F, size_t MaxDegree>
class Copyable<math::UnivariateEvaluations<F, MaxDegree>> {
 public:
  static bool WriteTo(const math::UnivariateEvaluations<F, MaxDegree>& evals,
                      Buffer* buffer) {
    return buffer->Write(evals.evaluations());
  }

  static bool ReadFrom(const Buffer& buffer,
                       math::UnivariateEvaluations<F, MaxDegree>* evals) {
    std::vector<F> evals_vec;
    if (!buffer.Read(&evals_vec)) return false;
    *evals = math::UnivariateEvaluations<F, MaxDegree>(evals_vec);
    return true;
  }

  static size_t EstimateSize(
      const math::UnivariateEvaluations<F, MaxDegree>& evals) {
    return base::EstimateSize(evals.evaluations());
  }
};

}  // namespace base
}  // namespace tachyon

#include "tachyon/math/polynomials/univariate/univariate_evaluations_ops.h"

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_H_
