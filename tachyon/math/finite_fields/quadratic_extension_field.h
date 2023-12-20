// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_QUADRATIC_EXTENSION_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_QUADRATIC_EXTENSION_FIELD_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/finite_fields/cyclotomic_multiplicative_subgroup.h"
#include "tachyon/math/geometry/point2.h"

namespace tachyon {
namespace math {

template <typename Derived>
class QuadraticExtensionField
    : public CyclotomicMultiplicativeSubgroup<Derived> {
 public:
  using Config = typename FiniteField<Derived>::Config;
  using BaseField = typename Config::BaseField;
  using MontgomeryTy = Point2<typename BaseField::MontgomeryTy>;

  constexpr QuadraticExtensionField() = default;
  constexpr QuadraticExtensionField(const BaseField& c0, const BaseField& c1)
      : c0_(c0), c1_(c1) {}
  constexpr QuadraticExtensionField(BaseField&& c0, BaseField&& c1)
      : c0_(std::move(c0)), c1_(std::move(c1)) {}

  constexpr static Derived Zero() {
    return {BaseField::Zero(), BaseField::Zero()};
  }

  constexpr static Derived One() {
    return {BaseField::One(), BaseField::Zero()};
  }

  static Derived Random() { return {BaseField::Random(), BaseField::Random()}; }

  constexpr static Derived FromMontgomery(const MontgomeryTy& mont) {
    return {BaseField::FromMontgomery(mont.x),
            BaseField::FromMontgomery(mont.y)};
  }

  constexpr bool IsZero() const { return c0_.IsZero() && c1_.IsZero(); }

  constexpr bool IsOne() const { return c0_.IsOne() && c1_.IsZero(); }

  constexpr static uint64_t ExtensionDegree() {
    return 2 * BaseField::ExtensionDegree();
  }

  constexpr Derived& ConjugateInPlace() {
    c1_.NegInPlace();
    return *static_cast<Derived*>(this);
  }

  // Norm of QuadraticExtensionField over |BaseField|:
  // |a.Norm() = a * a.Conjugate()|.
  // This simplifies to: |a.Norm() = a.c0² - Config::kNonResidue * a.c1²|.
  // This is alternatively expressed as |a.Norm() = aᵖ⁺¹|.
  constexpr BaseField Norm() const {
    return c0_.Square() - Config::MulByNonResidue(c1_.Square());
  }

  constexpr Derived& FrobeniusMapInPlace(uint64_t exponent) {
    c0_.FrobeniusMapInPlace(exponent);
    c1_.FrobeniusMapInPlace(exponent);
    c1_ *=
        Config::kFrobeniusCoeffs[exponent % Config::kDegreeOverBasePrimeField];
    return *static_cast<Derived*>(this);
  }

  constexpr MontgomeryTy ToMontgomery() const {
    return {c0_.ToMontgomery(), c1_.ToMontgomery()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", c0_.ToString(), c1_.ToString());
  }

  constexpr const BaseField& c0() const { return c0_; }
  constexpr const BaseField& c1() const { return c1_; }

  constexpr bool operator==(const Derived& other) const {
    return c0_ == other.c0_ && c1_ == other.c1_;
  }

  constexpr bool operator!=(const Derived& other) const {
    return c0_ != other.c0_ || c1_ != other.c1_;
  }

  constexpr bool operator<(const Derived& other) const {
    if (c1_ == other.c1_) return c0_ < other.c0_;
    return c1_ < other.c1_;
  }

  constexpr bool operator>(const Derived& other) const {
    if (c1_ == other.c1_) return c0_ > other.c0_;
    return c1_ > other.c1_;
  }

  constexpr bool operator<=(const Derived& other) const {
    if (c1_ == other.c1_) return c0_ <= other.c0_;
    return c1_ <= other.c1_;
  }

  constexpr bool operator>=(const Derived& other) const {
    if (c1_ == other.c1_) return c0_ >= other.c0_;
    return c1_ >= other.c1_;
  }

  // AdditiveSemigroup methods
  constexpr Derived& AddInPlace(const Derived& other) {
    c0_ += other.c0_;
    c1_ += other.c1_;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived& DoubleInPlace() {
    c0_.DoubleInPlace();
    c1_.DoubleInPlace();
    return *static_cast<Derived*>(this);
  }

  // AdditiveGroup methods
  constexpr Derived& SubInPlace(const Derived& other) {
    c0_ -= other.c0_;
    c1_ -= other.c1_;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived& NegInPlace() {
    c0_.NegInPlace();
    c1_.NegInPlace();
    return *static_cast<Derived*>(this);
  }

  // MultiplicativeSemigroup methods
  constexpr Derived& MulInPlace(const Derived& other) {
    // clang-format off
    // (c0, c1) * (other.c0, other.c1)
    //   = (c0 + c1 * x) * (other.c0 + other.c1 * x)
    //   = c0 * other.c0 + (c0 * other.c1 + c1 * other.c0) * x + c1 * other.c1 * x²
    //   = c0 * other.c0 + c1 * other.c1 * x² + (c0 * other.c1 + c1 * other.c0) * x
    //   = c0 * other.c0 + c1 * other.c1 * q + (c0 * other.c1 + c1 * other.c0) * x
    //   = (c0 * other.c0 + c1 * other.c1 * q, c0 * other.c0 +  c1 * other.c0)
    // Where q is Config::kNonResidue.
    // clang-format on
    if constexpr (ExtensionDegree() == 2) {
      BaseField c0;
      {
        BaseField lefts[] = {c0_, Config::MulByNonResidue(c1_)};
        BaseField rights[] = {other.c0_, other.c1_};
        c0 = BaseField::SumOfProductsSerial(lefts, rights);
      }
      BaseField c1;
      {
        BaseField lefts[] = {c0_, c1_};
        BaseField rights[] = {other.c1_, other.c0_};
        c1 = BaseField::SumOfProductsSerial(lefts, rights);
      }
      c0_ = std::move(c0);
      c1_ = std::move(c1);
    } else {
      // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
      // Karatsuba multiplication;
      // Guide to Pairing-based cryptography, Algorithm 5.16.
      // v0 = c0 * other.c0
      BaseField v0 = c0_ * other.c0_;
      // v1 = c1 * other.c1
      BaseField v1 = c1_ * other.c1_;

      // c1 = c0 + c1
      c1_ += c0_;
      // c1 = (c0 + c1) * (other.c0 + other.c1)
      // c1 = c0 * other.c0 + c0 * other.c1 + c1 * other.c0 + c1 * other.c1
      c1_ *= (other.c0_ + other.c1_);
      // c1 = c0 * other.c1 + c1 * other.c0 + c1 * other.c1
      c1_ -= v0;
      // c1 = c0 * other.c1 + c1 * other.c0
      c1_ -= v1;
      // c0 = c0 * other.c0 + q * c1 * other.c1
      c0_ = v0 + Config::MulByNonResidue(v1);
    }
    return *static_cast<Derived*>(this);
  }

  constexpr Derived& MulInPlace(const BaseField& element) {
    c0_ *= element;
    c1_ *= element;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived& SquareInPlace() {
    // (c0, c1)² = (c0 + c1 * x)²
    //            = c0² + 2 * c0 * c1 * x + c1² * x²
    //            = c0² + c1² * x² + 2 * c0 * c1 * x
    //            = c0² + c1² * q + 2 * c0 * c1 * x
    //            = (c0² + c1² * q, 2 * c0 * c1)
    // Where q is Config::kNonResidue.
    // When q = -1, we can re-use intermediate additions to improve performance.

    // v0 = c0 - c1
    BaseField v0 = c0_;
    v0 -= c1_;
    // v1 = c0 * c1
    BaseField v1 = c0_ * c1_;
    if constexpr (Config::kNonResidueIsMinusOne) {
      // When the non-residue is -1, we save 2 intermediate additions,
      // and use one fewer intermediate variable

      // v0 = (c0 - c1) * (c0 + c1)
      //    = c0² - c1²
      v0 *= (c0_ + c1_);

      // c0 = c0² - c1²
      c0_ = std::move(v0);
      // c1 = 2 * c0 * c1
      c1_ = v1.Double();
    } else {
      // v2 = c0 - q * c1
      BaseField v2 = c0_ - Config::MulByNonResidue(c1_);

      // v0 = (v0 * v2)
      // v0 = (c0 - c1) * (c0 - c1 * q)
      // v0 = c0² - c0 * c1 * q - c0 * c1 + c1² * q
      // v0 = c0² - (q + 1) * c0 * c1 + c1² * q
      // v0 = c0² + c1² * q - (q + 1) * c0 * c1
      v0 *= v2;

      // c0 = v0 + (q + 1) * c0 * c1
      // c0 = c0² + c1² * q - (q + 1) * c0 * c1 + (q + 1) * c0 * c1
      // c0 = c0² + c1² * q
      c0_ = std::move(v0);
      c0_ += v1;
      c0_ += Config::MulByNonResidue(v1);
      // c1 = 2 * c0 * c1
      c1_ = v1.Double();
    }
    return *static_cast<Derived*>(this);
  }

  // MultiplicativeGroup methods
  Derived& DivInPlace(const Derived& other) {
    return MulInPlace(other.Inverse());
  }

  constexpr Derived& InverseInPlace() {
    // NOTE(chokobole): CHECK(!IsZero()) is not a device code.
    // See https://github.com/kroma-network/tachyon/issues/76
    if (IsZero()) return *static_cast<Derived*>(this);
    // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
    // Guide to Pairing-based Cryptography, Algorithm 5.19.
    // v1 = c1²
    BaseField v1 = c1_.Square();
    // v0 = c0² - q * v1
    BaseField v0 = c0_.Square();
    v0 -= Config::MulByNonResidue(v1);

    v1 = v0.Inverse();
    c0_ *= v1;
    c1_ *= v1;
    c1_.NegInPlace();
    return *static_cast<Derived*>(this);
  }

  // CyclotomicMultiplicativeSubgroup methods
  constexpr Derived& FastCyclotomicInverseInPlace() {
    // As the multiplicative subgroup is of order p² - 1, the
    // only non-trivial cyclotomic subgroup is of order p + 1
    // Therefore, for any element in the cyclotomic subgroup, we have that
    // |xᵖ⁺¹ = 1|. Recall that |xᵖ⁺¹| in a quadratic extension
    // field is equal to the norm in the base field, so we have that
    // |x * x.Conjugate() = 1|. By uniqueness of inverses, for this subgroup,
    // |x.Inverse() = x.Conjugate()|.

    // NOTE(chokobole): CHECK(!IsZero()) is not a device code.
    // See https://github.com/kroma-network/tachyon/issues/76
    if (IsZero()) return *static_cast<Derived*>(this);

    return ConjugateInPlace();
  }

 protected:
  // c = c0_ + c1_ * X
  BaseField c0_;
  BaseField c1_;
};

template <
    typename BaseField, typename Derived,
    std::enable_if_t<std::is_same_v<BaseField, typename Derived::BaseField>>* =
        nullptr>
Derived operator*(const BaseField& element,
                  const QuadraticExtensionField<Derived>& f) {
  return static_cast<const Derived&>(f) * element;
}

}  // namespace math

namespace base {

template <typename Derived>
class Copyable<Derived, std::enable_if_t<std::is_base_of_v<
                            math::QuadraticExtensionField<Derived>, Derived>>> {
 public:
  static bool WriteTo(
      const math::QuadraticExtensionField<Derived>& quadratic_extension_field,
      Buffer* buffer) {
    return buffer->WriteMany(quadratic_extension_field.c0(),
                             quadratic_extension_field.c1());
  }

  static bool ReadFrom(
      const Buffer& buffer,
      math::QuadraticExtensionField<Derived>* quadratic_extension_field) {
    typename Derived::BaseField c0;
    typename Derived::BaseField c1;
    if (!buffer.ReadMany(&c0, &c1)) return false;

    *quadratic_extension_field =
        math::QuadraticExtensionField<Derived>(std::move(c0), std::move(c1));
    return true;
  }

  static size_t EstimateSize(
      const math::QuadraticExtensionField<Derived>& quadratic_extension_field) {
    return EstimateSize(quadratic_extension_field.c0()) +
           EstimateSize(quadratic_extension_field.c1());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_QUADRATIC_EXTENSION_FIELD_H_
