// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_

#include <stddef.h>

#include <string>
#include <type_traits>
#include <utility>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/polynomials/polynomial.h"
#include "tachyon/math/polynomials/univariate/univariate_dense_coefficients.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_forwards.h"
#include "tachyon/math/polynomials/univariate/univariate_sparse_coefficients.h"

namespace tachyon {
namespace math {

// UnivariatePolynomial represents a polynomial with a single variable.
// For example, 3x² + 2x + 1 is a univariate polynomial, while 3x²y + 2yz + 1 is
// not a univariate polynomial. The polynomial is represented as a vector of its
// coefficients. These coefficients are stored in an object which can be
// DenseCoefficients or SparseCoefficients.
template <typename Coefficients>
class UnivariatePolynomial final
    : public Polynomial<UnivariatePolynomial<Coefficients>> {
 public:
  constexpr static size_t kMaxSize = Coefficients::kMaxSize;

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

  constexpr static UnivariatePolynomial Random(size_t size) {
    return UnivariatePolynomial(Coefficients::Random(size));
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

  constexpr Field* operator[](size_t i) { return coefficients_[i]; }

  constexpr const Field* operator[](size_t i) const { return coefficients_[i]; }

  constexpr const Field* GetLeadingCoefficient() const {
    return coefficients_.GetLeadingCoefficient();
  }

  constexpr size_t Degree() const { return coefficients_.Degree(); }

  constexpr Field Evaluate(const Field& point) const {
    return coefficients_.Evaluate(point);
  }

  auto ToSparse() const {
    return internal::UnivariatePolynomialOp<Coefficients>::ToSparse(*this);
  }

  auto ToDense() const {
    return internal::UnivariatePolynomialOp<Coefficients>::ToDense(*this);
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
  friend class Radix2EvaluationDomain<Field, kMaxSize>;
  friend class MixedRadixEvaluationDomain<Field, kMaxSize>;

  // NOTE(chokobole): This doesn't call |RemoveHighDegreeZeros()| internally.
  // So when the returned evaluations is called with |IsZero()|, it returns
  // false. This is only used at |EvaluationDomain|.
  constexpr static UnivariatePolynomial UnsafeZero(size_t degree) {
    UnivariatePolynomial ret;
    ret.coefficients_ = Coefficients::UnsafeZero(degree);
    return ret;
  }

  Coefficients coefficients_;
};

template <typename F, size_t N>
using UnivariateDensePolynomial =
    UnivariatePolynomial<UnivariateDenseCoefficients<F, N>>;

template <typename F, size_t N>
using UnivariateSparsePolynomial =
    UnivariatePolynomial<UnivariateSparseCoefficients<F, N>>;

template <typename Coefficients>
class PolynomialTraits<UnivariatePolynomial<Coefficients>> {
 public:
  constexpr static bool kIsCoefficientForm = true;
};

}  // namespace math

namespace base {

template <typename Coefficients>
class Copyable<math::UnivariatePolynomial<Coefficients>> {
 public:
  static bool WriteTo(const math::UnivariatePolynomial<Coefficients>& poly,
                      Buffer* buffer) {
    return buffer->Write(poly.coefficients());
  }

  static bool ReadFrom(const Buffer& buffer,
                       math::UnivariatePolynomial<Coefficients>* poly) {
    Coefficients coeff;
    if (!buffer.Read(&coeff)) return false;
    *poly = math::UnivariatePolynomial<Coefficients>(coeff);
    return true;
  }

  static size_t EstimateSize(
      const math::UnivariatePolynomial<Coefficients>& poly) {
    return base::EstimateSize(poly.coefficients());
  }
};

}  // namespace base
}  // namespace tachyon

#include "tachyon/math/polynomials/univariate/univariate_polynomial_ops.h"

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_
