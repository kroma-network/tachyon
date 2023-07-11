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

  // AdditiveMonoid methods
  OPERATION_METHOD(Add)

  // AdditiveGroup methods
  OPERATION_METHOD(Sub)

  UnivariatePolynomial& NegativeInPlace() {
    return internal::UnivariatePolynomialOp<Coefficients>::NegativeInPlace(
        *this);
  }

  // MultiplicativeMonoid methods
  OPERATION_METHOD(Mul)

#undef OPERATION_METHOD

 private:
  friend class Polynomial<UnivariatePolynomial<Coefficients>>;
  friend class internal::UnivariatePolynomialOp<Coefficients>;

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

#include "tachyon/math/polynomials/univariate_polynomial_ops.h"

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_H_
