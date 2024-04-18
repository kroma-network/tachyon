#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_H_

#include <stddef.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tachyon/math/polynomials/multivariate/multivariate_sparse_coefficients.h"
#include "tachyon/math/polynomials/polynomial.h"

namespace tachyon::math {

// MultivariatePolynomial represents a polynomial with multiple variables.
// For example, 3x²y + 2yz + 1 is a multivariate polynomial with 3 variables.
template <typename Coefficients>
class MultivariatePolynomial final
    : public Polynomial<MultivariatePolynomial<Coefficients>> {
 public:
  constexpr static size_t kMaxDegree = Coefficients::kMaxDegree;

  using Field = typename Coefficients::Field;
  using Literal = typename Coefficients::Literal;
  using Point = std::vector<Field>;

  constexpr MultivariatePolynomial() = default;
  explicit constexpr MultivariatePolynomial(const Coefficients& coefficients)
      : coefficients_(coefficients) {}
  explicit constexpr MultivariatePolynomial(Coefficients&& coefficients)
      : coefficients_(std::move(coefficients)) {}

  constexpr static MultivariatePolynomial Zero() {
    return MultivariatePolynomial(Coefficients::Zero());
  }

  constexpr static MultivariatePolynomial One() {
    return MultivariatePolynomial(Coefficients::One());
  }

  constexpr static MultivariatePolynomial Random(size_t arity, size_t degree) {
    return MultivariatePolynomial(Coefficients::Random(arity, degree));
  }

  constexpr static bool IsCoefficientForm() { return true; }

  constexpr static bool IsEvaluationForm() { return false; }

  constexpr bool IsZero() const { return coefficients_.IsZero(); }

  constexpr bool IsOne() const { return coefficients_.IsOne(); }

  constexpr const Coefficients& coefficients() const { return coefficients_; }

  constexpr bool operator==(const MultivariatePolynomial& other) const {
    return coefficients_ == other.coefficients_;
  }

  constexpr bool operator!=(const MultivariatePolynomial& other) const {
    return !operator==(other);
  }

  // Returns a reference to the coefficient for the given |i| if it exists.
  // Otherwise, it terminates the program.
  constexpr Field& at(const Literal& literal) {
    return coefficients_.at(literal);
  }

  // Returns a reference to the coefficient for the given |i| if it exists.
  // Otherwise, returns a reference to the |Field::Zero()|.
  constexpr const Field& at(const Literal& literal) const {
    return coefficients_.at(literal);
  }

  // Returns a reference to the coefficient for the given |i| if it exists.
  // Otherwise, returns a reference to the |Field::Zero()|.
  constexpr const Field& operator[](const Literal& literal) const {
    return coefficients_[literal];
  }

  // Returns a reference to the leading coefficient if it exists.
  // Otherwise, returns a reference to the |Field::Zero()|.
  constexpr const Field& GetLeadingCoefficient() const {
    return coefficients_.GetLeadingCoefficient();
  }

  constexpr size_t Degree() const { return coefficients_.Degree(); }

  constexpr Field Evaluate(const Point& point) const {
    return coefficients_.Evaluate(point);
  }

  std::string ToString() const { return coefficients_.ToString(); }

#define OPERATION_METHOD(Name)                                              \
  template <typename Coefficients2,                                         \
            std::enable_if_t<internal::SupportsPoly##Name<                  \
                Coefficients, MultivariatePolynomial<Coefficients>,         \
                MultivariatePolynomial<Coefficients2>>::value>* = nullptr>  \
  constexpr auto Name(const MultivariatePolynomial<Coefficients2>& other)   \
      const {                                                               \
    return internal::MultivariatePolynomialOp<Coefficients>::Name(*this,    \
                                                                  other);   \
  }                                                                         \
                                                                            \
  template <typename Coefficients2,                                         \
            std::enable_if_t<internal::SupportsPoly##Name##InPlace<         \
                Coefficients, MultivariatePolynomial<Coefficients>,         \
                MultivariatePolynomial<Coefficients2>>::value>* = nullptr>  \
  constexpr auto& Name##InPlace(                                            \
      const MultivariatePolynomial<Coefficients2>& other) {                 \
    return internal::MultivariatePolynomialOp<Coefficients>::Name##InPlace( \
        *this, other);                                                      \
  }

  // AdditiveSemigroup methods
  OPERATION_METHOD(Add)

  // AdditiveGroup methods
  OPERATION_METHOD(Sub)

  MultivariatePolynomial Negate() const {
    return internal::MultivariatePolynomialOp<Coefficients>::Negate(*this);
  }

  MultivariatePolynomial& NegateInPlace() {
    return internal::MultivariatePolynomialOp<Coefficients>::NegateInPlace(
        *this);
  }
#undef OPERATION_METHOD

 private:
  friend class internal::MultivariatePolynomialOp<Coefficients>;

  Coefficients coefficients_;
};

template <typename F, size_t MaxDegree>
using MultivariateSparsePolynomial =
    MultivariatePolynomial<MultivariateSparseCoefficients<F, MaxDegree>>;

template <typename Coefficients>
class PolynomialTraits<MultivariatePolynomial<Coefficients>> {
 public:
  constexpr static bool kIsCoefficientForm = true;
};

}  // namespace tachyon::math

#include "tachyon/math/polynomials/multivariate/multivariate_polynomial_ops.h"

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTIVARIATE_POLYNOMIAL_H_
