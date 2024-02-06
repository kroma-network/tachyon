// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"

#include "tachyon/math/polynomials/multivariate/multilinear_dense_evaluations.h"
#include "tachyon/math/polynomials/polynomial.h"

namespace tachyon::math {

// MultilinearExtension represents a multilinear polynomial in evaluation form.
// Unlike UnivariateEvaluations, its evaluation domain is fixed to {0, 1}ᵏ (i.e.
// Boolean hypercube).
template <typename Evaluations>
class MultilinearExtension final
    : public Polynomial<MultilinearExtension<Evaluations>> {
 public:
  using Field = typename Evaluations::Field;
  using Point = std::vector<Field>;

  constexpr MultilinearExtension() = default;
  constexpr explicit MultilinearExtension(const Evaluations& evaluations)
      : evaluations_(evaluations) {}
  constexpr explicit MultilinearExtension(Evaluations&& evaluations)
      : evaluations_(std::move(evaluations)) {}

  constexpr static bool IsCoefficientForm() { return false; }

  constexpr static bool IsEvaluationForm() { return true; }

  constexpr static MultilinearExtension Zero() {
    return MultilinearExtension(Evaluations::Zero());
  }

  constexpr static MultilinearExtension One(size_t degree) {
    return MultilinearExtension(Evaluations::One(degree));
  }

  constexpr static MultilinearExtension Random(size_t degree) {
    return MultilinearExtension(Evaluations::Random(degree));
  }

  constexpr bool IsZero() const { return evaluations_.IsZero(); }

  constexpr bool IsOne() const { return evaluations_.IsOne(); }

  const Evaluations& evaluations() const { return evaluations_; }

  constexpr bool operator==(const MultilinearExtension& other) const {
    return evaluations_ == other.evaluations_;
  }

  constexpr bool operator!=(const MultilinearExtension& other) const {
    return !operator==(other);
  }

  constexpr Field* operator[](size_t i) { return evaluations_[i]; }

  constexpr const Field* operator[](size_t i) const { return evaluations_[i]; }

  constexpr size_t Degree() const { return evaluations_.Degree(); }

  // Evaluate a polynomial at the specified |point|. The |point| is a vector in
  // {0, 1}ᵏ in little-endian form. If the size of |point| is less than the
  // degree of the polynomial, the remaining components of |point| are assumed
  // to be zeros. For example:

  //   P(x₀, x₁) = 1(1 - x₀)(1 - x₁) + 3x₀(1 - x₁) + 5(1 - x₀)x₁ + 2x₀x₁
  //   P(0, 0) = 1
  //   P(0, 1) = 5
  //   P(1, 0) = 3
  //   P(1, 1) = 2

  // In this context, {x₀, x₁} corresponds to the components of the |point|.
  constexpr Field Evaluate(const Point& point) const {
    return evaluations_.Evaluate(point);
  }

  auto ToDense() const {
    return internal::MultilinearExtensionOp<Evaluations>::ToDense(*this);
  }

  std::string ToString() const { return evaluations_.ToString(); }

#define OPERATION_METHOD(Name)                                                 \
  template <typename Evaluations2,                                             \
            std::enable_if_t<internal::SupportsPoly##Name<                     \
                Evaluations, MultilinearExtension<Evaluations>,                \
                MultilinearExtension<Evaluations2>>::value>* = nullptr>        \
  constexpr auto Name(const MultilinearExtension<Evaluations2>& other) const { \
    return internal::MultilinearExtensionOp<Evaluations>::Name(*this, other);  \
  }                                                                            \
                                                                               \
  template <typename Evaluations2,                                             \
            std::enable_if_t<internal::SupportsPoly##Name##InPlace<            \
                Evaluations, MultilinearExtension<Evaluations>,                \
                MultilinearExtension<Evaluations2>>::value>* = nullptr>        \
  constexpr auto& Name##InPlace(                                               \
      const MultilinearExtension<Evaluations2>& other) {                       \
    return internal::MultilinearExtensionOp<Evaluations>::Name##InPlace(       \
        *this, other);                                                         \
  }

  // AdditiveSemigroup methods
  OPERATION_METHOD(Add)

  // AdditiveGroup methods
  OPERATION_METHOD(Sub)

  MultilinearExtension& NegInPlace() {
    return internal::MultilinearExtensionOp<Evaluations>::NegInPlace(*this);
  }

  // MultiplicativeSemigroup methods
  OPERATION_METHOD(Mul)

  OPERATION_METHOD(Div)

  constexpr MultilinearExtension operator/(
      const MultilinearExtension& other) const {
    MultilinearExtension poly = *this;
    return poly.DivInPlace(other);
  }

  constexpr MultilinearExtension& operator/=(
      const MultilinearExtension& other) {
    return DivInPlace(other);
  }

 private:
  friend class internal::MultilinearExtensionOp<
      MultilinearDenseEvaluations<Field, Evaluations::kMaxDegree>>;

  Evaluations evaluations_;
};

template <typename F, size_t MaxDegree>
using MultilinearDenseExtension =
    MultilinearExtension<MultilinearDenseEvaluations<F, MaxDegree>>;

template <typename Evaluations>
class PolynomialTraits<MultilinearExtension<Evaluations>> {
 public:
  constexpr static bool kIsEvaluationForm = true;
};

template <typename H, typename Evaluations>
H AbslHashValue(H h, const MultilinearExtension<Evaluations>& mle) {
  return H::combine(std::move(h), mle.evaluations());
}

}  // namespace tachyon::math

#include "tachyon/math/polynomials/multivariate/multilinear_extension_ops.h"

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_MULTILINEAR_EXTENSION_H_
