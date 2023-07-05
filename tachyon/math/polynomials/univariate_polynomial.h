#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_H_

#include <stddef.h>

#include <type_traits>

#include "tachyon/base/logging.h"
#include "tachyon/math/polynomials/dense_coefficients.h"
#include "tachyon/math/polynomials/polynomial.h"
#include "tachyon/math/polynomials/sparse_coefficients.h"

namespace tachyon {
namespace math {

template <typename Coefficients>
class UnivariatePolynomial
    : public Polynomial<UnivariatePolynomial<Coefficients>> {
 public:
  constexpr static const size_t MAX_DEGREE = Coefficients::MAX_DEGREE;

  using Field = typename Coefficients::Field;

  constexpr UnivariatePolynomial() = default;
  constexpr explicit UnivariatePolynomial(const Coefficients& coefficients)
      : coefficients_(coefficients) {}
  constexpr explicit UnivariatePolynomial(Coefficients&& coefficients)
      : coefficients_(std::move(coefficients)) {}

  const Coefficients& coefficients() const { return coefficients_; }

  constexpr Field* operator[](size_t i) { return coefficients_.Get(i); }

  constexpr const Field* operator[](size_t i) const {
    return coefficients_.Get(i);
  }

  std::string ToString() const { return coefficients_.ToString(); }

 private:
  friend class Polynomial<UnivariatePolynomial<Coefficients>>;

  // Polynomial methods
  constexpr size_t DoDegree() const { return coefficients_.Degree(); }

  constexpr Field DoEvaluate(const Field& point) const {
    return coefficients_.Evaluate(point);
  }

  Coefficients coefficients_;
};

template <typename Coefficients>
std::ostream& operator<<(std::ostream& os,
                         const UnivariatePolynomial<Coefficients>& p) {
  return os << p.ToString();
}

template <typename Coefficients>
class CoefficientsTraits<UnivariatePolynomial<Coefficients>> {
 public:
  using Field = typename Coefficients::Field;
};

template <typename F, size_t MAX_DEGREE>
using DenseUnivariatePolynomial =
    UnivariatePolynomial<DenseCoefficients<F, MAX_DEGREE>>;

template <typename F, size_t MAX_DEGREE>
using SparseUnivariatePolynomial =
    UnivariatePolynomial<SparseCoefficients<F, MAX_DEGREE>>;

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_H_
