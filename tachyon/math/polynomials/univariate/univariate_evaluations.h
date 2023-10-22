// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/polynomials/polynomial.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_forwards.h"

namespace tachyon::math {

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

 private:
  friend class Radix2EvaluationDomain<F, MaxDegree>;
  friend class MixedRadixEvaluationDomain<F, kMaxDegree>;

  std::vector<F> evaluations_;
};

template <typename F, size_t MaxDegree>
class PolynomialTraits<UnivariateEvaluations<F, MaxDegree>> {
 public:
  constexpr static bool kIsEvaluationForm = true;
};

}  // namespace tachyon::math

#include "tachyon/math/polynomials/univariate/univariate_polynomial_ops.h"

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATIONS_H_
