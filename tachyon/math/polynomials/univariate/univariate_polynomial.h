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

#include "absl/hash/hash.h"

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
  constexpr static size_t kMaxDegree = Coefficients::kMaxDegree;

  using Field = typename Coefficients::Field;
  using Point = Field;

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

  // Return a vanishing polynomial according to the given |roots|.
  template <typename ContainerTy>
  constexpr static UnivariatePolynomial FromRoots(const ContainerTy& roots) {
    using DenseCoeffs = UnivariateDenseCoefficients<Field, kMaxDegree>;
    if constexpr (std::is_same_v<Coefficients, DenseCoeffs>) {
      return UnivariatePolynomial(Coefficients::FromRoots(roots));
    } else {
      using DensePoly = UnivariatePolynomial<DenseCoeffs>;
      return DensePoly(DenseCoeffs::FromRoots(roots)).ToSparse();
    }
  }

  // NOTE(chokobole): For a performance reason, I would recommend the
  // |LinearizeInPlace()| if possible.
  template <typename ContainerTy>
  constexpr static UnivariatePolynomial Linearize(const ContainerTy& polys,
                                                  const Field& r) {
    CHECK(!polys.empty());
    UnivariatePolynomial ret = polys[polys.size() - 1];
    if (polys.size() > 1) {
      for (size_t i = polys.size() - 2; i != SIZE_MAX; --i) {
        ret *= r;
        ret += polys[i];
      }
    }
    return ret;
  }

  // NOTE(chokobole): This gives more performant result than |Linearize()|.
  //
  // clang-format off
  //
  // In normal case, you can linearize polynomials as follows:
  //
  //   const std::vector<UnivariatePolynomial> polys = {...};
  //   UnivariatePolynomial ret = UnivariatePolynomial::Linearize(polys, Field::Random());
  //
  // If you can do like below, you can save additional allocation cost.
  //
  //   // Note that |polys| are going to be changed.
  //   std::vector<UnivariatePolynomial> polys = {...};
  //   UnivariatePolynomial& ret = UnivariatePolynomial::LinearizeInPlace(polys, Field::Random());
  //
  // clang-format off
  template <typename ContainerTy>
  constexpr static UnivariatePolynomial& LinearizeInPlace(ContainerTy& polys, const Field& r) {
    CHECK(!polys.empty());
    UnivariatePolynomial& ret = polys[polys.size() - 1];
    if (polys.size() > 1) {
      for (size_t i = polys.size() - 2; i != SIZE_MAX; --i) {
        ret *= r;
        ret += polys[i];
      }
    }
    return ret;
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

  constexpr const size_t NumElements() const {
    return coefficients_.NumElements();
  }

  constexpr Field Evaluate(const Point& point) const {
    return coefficients_.Evaluate(point);
  }

  template <typename ContainerTy>
  constexpr static Field EvaluateVanishingPolyByRoots(const ContainerTy& roots,
                                                      const Field& point) {
    return std::accumulate(roots.begin(), roots.end(), Field::One(),
                           [point](Field& acc, const Field& root) {
                             return acc *= (point - root);
                           });
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

  UnivariatePolynomial& MulInPlace(const Field& scalar) {
    return internal::UnivariatePolynomialOp<Coefficients>::MulInPlace(*this,
                                                                      scalar);
  }

  OPERATION_METHOD(Div)
  OPERATION_METHOD(Mod)

#undef OPERATION_METHOD

  UnivariatePolynomial operator/(const Field& scalar) const {
    UnivariatePolynomial poly = *this;
    poly /= scalar;
    return poly;
  }

  UnivariatePolynomial& operator/=(const Field& scalar) {
    return internal::UnivariatePolynomialOp<Coefficients>::DivInPlace(*this,
                                                                      scalar);
  }

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
  friend class MixedRadixEvaluationDomain<Field, kMaxDegree>;

  // NOTE(chokobole): This doesn't call |RemoveHighDegreeZeros()| internally.
  // So when the returned evaluations is called with |IsZero()|, it returns
  // false. So please use it carefully!
  constexpr static UnivariatePolynomial UnsafeZero(size_t degree) {
    UnivariatePolynomial ret;
    ret.coefficients_ = Coefficients::UnsafeZero(degree);
    return ret;
  }

  Coefficients coefficients_;
};

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

template <typename H, typename Coefficients>
H AbslHashValue(H h, const UnivariatePolynomial<Coefficients>& poly) {
  return H::combine(std::move(h), poly.coefficients());
}

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
