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

  constexpr UnivariateEvaluations() = default;
  constexpr explicit UnivariateEvaluations(const std::vector<F>& evaluations)
      : evaluations_(evaluations) {
    CHECK_LE(Degree(), MaxDegree);
    RemoveHighDegreeZeros();
  }
  constexpr explicit UnivariateEvaluations(std::vector<F>&& evaluations)
      : evaluations_(std::move(evaluations)) {
    CHECK_LE(Degree(), MaxDegree);
    RemoveHighDegreeZeros();
  }

  constexpr static bool IsCoefficientForm() { return false; }

  constexpr static bool IsEvaluationForm() { return true; }

  constexpr static UnivariateEvaluations Zero() {
    return UnivariateEvaluations();
  }

  constexpr static UnivariateEvaluations One() {
    return UnivariateEvaluations({F::One()});
  }

  constexpr static UnivariateEvaluations Random(size_t degree) {
    return UnivariateDenseCoefficients(
        base::CreateVector(degree + 1, []() { return F::Random(); }));
  }

  const std::vector<F>& evaluations() const { return evaluations_; }

  constexpr bool IsZero() const { return evaluations_.empty(); }

  constexpr bool IsOne() const {
    return evaluations_.size() == 1 && evaluations_[0].IsOne();
  }

  constexpr size_t Degree() const {
    if (IsZero()) return 0;
    return evaluations_.size() - 1;
  }

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

  void RemoveHighDegreeZeros() {
    while (!IsZero()) {
      if (evaluations_.back().IsZero()) {
        evaluations_.pop_back();
      } else {
        break;
      }
    }
  }

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
