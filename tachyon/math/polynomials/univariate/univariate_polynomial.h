// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/json/json.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/parallelize.h"
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
template <typename _Coefficients>
class UnivariatePolynomial final
    : public Polynomial<UnivariatePolynomial<_Coefficients>> {
 public:
  using Coefficients = _Coefficients;

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
  template <typename Container>
  constexpr static UnivariatePolynomial FromRoots(const Container& roots) {
    using DenseCoeffs = UnivariateDenseCoefficients<Field, kMaxDegree>;
    if constexpr (std::is_same_v<Coefficients, DenseCoeffs>) {
      return UnivariatePolynomial(Coefficients::FromRoots(roots));
    } else {
      using DensePoly = UnivariatePolynomial<DenseCoeffs>;
      return DensePoly(DenseCoeffs::FromRoots(roots)).ToSparse();
    }
  }

  constexpr bool IsZero() const { return coefficients_.IsZero(); }

  constexpr bool IsOne() const { return coefficients_.IsOne(); }

  const Coefficients& coefficients() const { return coefficients_; }
  Coefficients& coefficients() { return coefficients_; }

  Coefficients&& TakeCoefficients() && { return std::move(coefficients_); }

  constexpr bool operator==(const UnivariatePolynomial& other) const {
    return coefficients_ == other.coefficients_;
  }

  constexpr bool operator!=(const UnivariatePolynomial& other) const {
    return !operator==(other);
  }

  // Returns a reference to the coefficient for the given |i| if it exists.
  // Otherwise, it terminates the program.
  constexpr Field& at(size_t i) { return coefficients_.at(i); }

  // Returns a reference to the coefficient for the given |i| if it exists.
  // Otherwise, returns a reference to the |Field::Zero()|.
  constexpr const Field& at(size_t i) const { return coefficients_.at(i); }

  // Returns a reference to the coefficient for the given |i| if it exists.
  // Otherwise, returns a reference to the |Field::Zero()|.
  constexpr const Field& operator[](size_t i) const { return coefficients_[i]; }

  // Returns a reference to the leading coefficient if it exists.
  // Otherwise, returns a reference to the |Field::Zero()|.
  constexpr const Field& GetLeadingCoefficient() const {
    return coefficients_.GetLeadingCoefficient();
  }

  constexpr size_t Degree() const { return coefficients_.Degree(); }

  constexpr const size_t NumElements() const {
    return coefficients_.NumElements();
  }

  constexpr Field Evaluate(const Point& point) const {
    return coefficients_.Evaluate(point);
  }

  template <typename Container>
  constexpr static Field EvaluateVanishingPolyByRootsSerial(
      const Container& roots, const Field& point) {
    return std::accumulate(roots.begin(), roots.end(), Field::One(),
                           [point](Field& acc, const Field& root) {
                             return acc *= (point - root);
                           });
  }

  template <typename Container>
  constexpr static Field EvaluateVanishingPolyByRoots(const Container& roots,
                                                      const Field& point) {
    std::vector<Field> products =
        base::ParallelizeMap(roots, [&point](absl::Span<const Field> chunk) {
          return EvaluateVanishingPolyByRootsSerial(chunk, point);
        });
    return std::accumulate(products.begin(), products.end(), Field::One(),
                           std::multiplies<>());
  }

  // Return a polynomial where the original polynomial reduces its degree
  // by categorizing coefficients into even and odd degrees,
  // multiplying either set of coefficients by a specified random field |r|,
  // and summing them together.
  template <bool MulRandomWithEvens>
  constexpr UnivariatePolynomial Fold(const Field& r) const {
    return UnivariatePolynomial(
        coefficients_.template Fold<MulRandomWithEvens>(r));
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

  UnivariatePolynomial Mul(const Field& scalar) const {
    return internal::UnivariatePolynomialOp<Coefficients>::Mul(*this, scalar);
  }

  UnivariatePolynomial& MulInPlace(const Field& scalar) {
    return internal::UnivariatePolynomialOp<Coefficients>::MulInPlace(*this,
                                                                      scalar);
  }

  OPERATION_METHOD(Div)
  OPERATION_METHOD(Mod)

#undef OPERATION_METHOD

  UnivariatePolynomial operator/(const Field& scalar) const {
    return internal::UnivariatePolynomialOp<Coefficients>::Div(*this, scalar);
  }

  UnivariatePolynomial& operator/=(const Field& scalar) {
    return internal::UnivariatePolynomialOp<Coefficients>::DivInPlace(*this,
                                                                      scalar);
  }

  template <typename Coefficients2>
  constexpr auto operator/(
      const UnivariatePolynomial<Coefficients2>& other) const {
    return Div(other);
  }

  template <typename Coefficients2>
  constexpr auto& operator/=(const UnivariatePolynomial<Coefficients2>& other) {
    return DivInPlace(other);
  }

  template <typename Coefficients2>
  constexpr auto operator%(
      const UnivariatePolynomial<Coefficients2>& other) const {
    return Mod(other);
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
  friend class UnivariateEvaluationDomain<Field, kMaxDegree>;
  friend class Radix2EvaluationDomain<Field, kMaxDegree>;
  friend class MixedRadixEvaluationDomain<Field, kMaxDegree>;

  // NOTE(chokobole): This doesn't call |RemoveHighDegreeZeros()| internally.
  // So when the returned instance of |UnivariatePolynomial| is called with
  // |IsZero()|, it returns false. So please use it carefully!
  constexpr static UnivariatePolynomial Zero(size_t degree) {
    UnivariatePolynomial ret;
    ret.coefficients_ = Coefficients::Zero(degree);
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

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
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

template <typename F, size_t MaxDegree>
class RapidJsonValueConverter<math::UnivariateDensePolynomial<F, MaxDegree>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(
      const math::UnivariateDensePolynomial<F, MaxDegree>& value,
      Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "coefficients", value.coefficients(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::UnivariateDensePolynomial<F, MaxDegree>* value,
                 std::string* error) {
    math::UnivariateDenseCoefficients<F, MaxDegree> dense_coeffs;
    if (!ParseJsonElement(json_value, "coefficients", &dense_coeffs, error))
      return false;
    *value = math::UnivariatePolynomial(std::move(dense_coeffs));
    return true;
  }
};

template <typename F, size_t MaxDegree>
class RapidJsonValueConverter<math::UnivariateSparsePolynomial<F, MaxDegree>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(
      const math::UnivariateSparsePolynomial<F, MaxDegree>& value,
      Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "coefficients", value.coefficients(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::UnivariateSparsePolynomial<F, MaxDegree>* value,
                 std::string* error) {
    math::UnivariateSparseCoefficients<F, MaxDegree> sparse_coeffs;
    if (!ParseJsonElement(json_value, "coefficients", &sparse_coeffs, error))
      return false;
    *value = math::UnivariatePolynomial(std::move(sparse_coeffs));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#include "tachyon/math/polynomials/univariate/univariate_polynomial_ops.h"

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_POLYNOMIAL_H_
