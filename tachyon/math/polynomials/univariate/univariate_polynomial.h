#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_

#include <stddef.h>

#include <string>
#include <type_traits>
#include <utility>

#include "tachyon/base/logging.h"
#include "tachyon/math/polynomials/polynomial.h"
#include "tachyon/math/polynomials/univariate/univariate_dense_coefficients.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_forwards.h"
#include "tachyon/math/polynomials/univariate/univariate_sparse_coefficients.h"

namespace tachyon::math {

// UnivariatePolynomial represents a polynomial with a single variable.
// For example, 3x² + 2x + 1 is a univariate polynomial, while 3x²y + 2yz + 1 is
// not a univariate polynomial. The polynomial is represented as a vector of its
// coefficients. These coefficients are stored in an object which can be
// DenseCoefficients or SparseCoefficients.
template <typename Coefficients>
class UnivariatePolynomial final
    : public Polynomial<UnivariatePolynomial<Coefficients>> {
 public:
  constexpr static const size_t kMaxDegree = Coefficients::kMaxDegree;

  using Field = typename Coefficients::Field;

  constexpr UnivariatePolynomial() = default;
  constexpr explicit UnivariatePolynomial(const Coefficients& coefficients)
      : coefficients_(coefficients) {}
  constexpr explicit UnivariatePolynomial(Coefficients&& coefficients)
      : coefficients_(std::move(coefficients)) {}

  constexpr static bool IsCoefficientForm() { return true; }

  constexpr static bool IsEvaluationForm() { return false; }

  constexpr static UnivariatePolynomial Zero() {
    return UnivariatePolynomial(Coefficients::Zero());
  }

  constexpr static UnivariatePolynomial One() {
    return UnivariatePolynomial(Coefficients::One());
  }

  constexpr static UnivariatePolynomial Random(size_t degree) {
    return UnivariatePolynomial(Coefficients::Random(degree));
  }

  constexpr bool IsZero() const { return coefficients_.IsZero(); }

  constexpr bool IsOne() const { return coefficients_.IsOne(); }

  const Coefficients& coefficients() const { return coefficients_; }

  constexpr bool operator==(const UnivariatePolynomial& other) const {
    return coefficients_ == other.coefficients_;
  }

  constexpr bool operator!=(const UnivariatePolynomial& other) const {
    return !operator==(other);
  }

  constexpr Field* operator[](size_t i) { return coefficients_.Get(i); }

  constexpr const Field* operator[](size_t i) const {
    return coefficients_.Get(i);
  }

  constexpr const Field* GetLeadingCoefficient() const {
    return coefficients_.GetLeadingCoefficient();
  }

  constexpr size_t Degree() const { return coefficients_.Degree(); }

  constexpr Field Evaluate(const Field& point) const {
    return coefficients_.Evaluate(point);
  }

  auto ToSparse() const {
    return internal::UnivariatePolynomialOp<Coefficients>::ToSparsePolynomial(
        *this);
  }

  auto ToDense() const {
    return internal::UnivariatePolynomialOp<Coefficients>::ToDensePolynomial(
        *this);
  }

  std::string ToString() const { return coefficients_.ToString(); }

#define OPERATION_METHOD(Name)                                                 \
  template <typename Coefficients2,                                            \
            std::enable_if_t<internal::SupportsPoly##Name<                     \
                Coefficients, UnivariatePolynomial<Coefficients>,              \
                UnivariatePolynomial<Coefficients2>>::value>* = nullptr>       \
  constexpr auto Name(const UnivariatePolynomial<Coefficients2>& other)        \
      const {                                                                  \
    return internal::UnivariatePolynomialOp<Coefficients>::Name(*this, other); \
  }                                                                            \
                                                                               \
  template <typename Coefficients2,                                            \
            std::enable_if_t<internal::SupportsPoly##Name##InPlace<            \
                Coefficients, UnivariatePolynomial<Coefficients>,              \
                UnivariatePolynomial<Coefficients2>>::value>* = nullptr>       \
  constexpr auto& Name##InPlace(                                               \
      const UnivariatePolynomial<Coefficients2>& other) {                      \
    return internal::UnivariatePolynomialOp<Coefficients>::Name##InPlace(      \
        *this, other);                                                         \
  }

  // AdditiveSemigroup methods
  OPERATION_METHOD(Add)

  // AdditiveGroup methods
  OPERATION_METHOD(Sub)

  UnivariatePolynomial& NegInPlace() {
    return internal::UnivariatePolynomialOp<Coefficients>::NegInPlace(*this);
  }

  // MultiplicativeSemigroup methods
  OPERATION_METHOD(Mul)

  OPERATION_METHOD(Div)
  OPERATION_METHOD(Mod)

#undef OPERATION_METHOD

  template <typename Coefficients2>
  constexpr auto operator/(
      const UnivariatePolynomial<Coefficients2>& other) const {
    if constexpr (internal::SupportsDiv<
                      UnivariatePolynomial,
                      UnivariatePolynomial<Coefficients2>>::value) {
      return Div(other);
    } else {
      UnivariatePolynomial poly = *this;
      return poly.DivInPlace(other);
    }
  }

  template <typename Coefficients2>
  constexpr auto& operator/=(const UnivariatePolynomial<Coefficients2>& other) {
    return DivInPlace(other);
  }

  template <typename Coefficients2>
  constexpr auto operator%(
      const UnivariatePolynomial<Coefficients2>& other) const {
    if constexpr (internal::SupportsMod<
                      UnivariatePolynomial,
                      UnivariatePolynomial<Coefficients2>>::value) {
      return Mod(other);
    } else {
      UnivariatePolynomial poly = *this;
      return poly.ModInPlace(other);
    }
  }

  template <typename Coefficients2>
  constexpr auto& operator%=(const UnivariatePolynomial<Coefficients2>& other) {
    return ModInPlace(other);
  }

  template <typename Coefficients2>
  constexpr auto DivMod(
      const UnivariatePolynomial<Coefficients2>& other) const {
    return internal::UnivariatePolynomialOp<Coefficients>::DivMod(*this, other);
  }

 private:
  friend class internal::UnivariatePolynomialOp<Coefficients>;
  friend class Radix2EvaluationDomain<Field, kMaxDegree>;

  Coefficients coefficients_;
};

template <typename Coefficients>
std::ostream& operator<<(std::ostream& os,
                         const UnivariatePolynomial<Coefficients>& p) {
  return os << p.ToString();
}

template <typename F, size_t MaxDegree>
using UnivariateDensePolynomial =
    UnivariatePolynomial<UnivariateDenseCoefficients<F, MaxDegree>>;

template <typename F, size_t MaxDegree>
using UnivariateSparsePolynomial =
    UnivariatePolynomial<UnivariateSparseCoefficients<F, MaxDegree>>;

template <typename Coefficients>
class PolynomialTraits<UnivariatePolynomial<Coefficients>> {
 public:
  constexpr static bool kIsCoefficientForm = true;
};

}  // namespace tachyon::math

#include "tachyon/math/polynomials/univariate/univariate_polynomial_ops.h"

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_
