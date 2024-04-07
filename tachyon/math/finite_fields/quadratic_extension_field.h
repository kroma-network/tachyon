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
#include "tachyon/base/json/json.h"
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

  constexpr Derived Conjugate() const {
    return {
        c0_,
        -c1_,
    };
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

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1)", c0_.ToHexString(pad_zero),
                            c1_.ToHexString(pad_zero));
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
  constexpr Derived Add(const Derived& other) const {
    return {
        c0_ + other.c0_,
        c1_ + other.c1_,
    };
  }

  constexpr Derived& AddInPlace(const Derived& other) {
    c0_ += other.c0_;
    c1_ += other.c1_;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived DoDouble() const {
    return {
        c0_.Double(),
        c1_.Double(),
    };
  }

  constexpr Derived& DoDoubleInPlace() {
    c0_.DoubleInPlace();
    c1_.DoubleInPlace();
    return *static_cast<Derived*>(this);
  }

  // AdditiveGroup methods
  constexpr Derived Sub(const Derived& other) const {
    return {
        c0_ - other.c0_,
        c1_ - other.c1_,
    };
  }

  constexpr Derived& SubInPlace(const Derived& other) {
    c0_ -= other.c0_;
    c1_ -= other.c1_;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived Negative() const {
    return {
        -c0_,
        -c1_,
    };
  }

  constexpr Derived& NegInPlace() {
    c0_.NegInPlace();
    c1_.NegInPlace();
    return *static_cast<Derived*>(this);
  }

  // MultiplicativeSemigroup methods
  constexpr Derived Mul(const Derived& other) const {
    Derived ret;
    DoMul(*static_cast<const Derived*>(this), other, ret);
    return ret;
  }

  constexpr Derived& MulInPlace(const Derived& other) {
    DoMul(*static_cast<const Derived*>(this), other,
          *static_cast<Derived*>(this));
    return *static_cast<Derived*>(this);
  }

  constexpr Derived Mul(const BaseField& element) const {
    return {
        c0_ * element,
        c1_ * element,
    };
  }

  constexpr Derived& MulInPlace(const BaseField& element) {
    c0_ *= element;
    c1_ *= element;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived DoSquare() const {
    Derived ret;
    DoSquareImpl(*static_cast<const Derived*>(this), ret);
    return ret;
  }

  constexpr Derived& DoSquareInPlace() {
    DoSquareImpl(*static_cast<const Derived*>(this),
                 *static_cast<Derived*>(this));
    return *static_cast<Derived*>(this);
  }

  // MultiplicativeGroup methods
  constexpr Derived Inverse() const {
    Derived ret;
    DoInverse(*static_cast<const Derived*>(this), ret);
    return ret;
  }

  constexpr Derived& InverseInPlace() {
    DoInverse(*static_cast<const Derived*>(this), *static_cast<Derived*>(this));
    return *static_cast<Derived*>(this);
  }

  // CyclotomicMultiplicativeSubgroup methods
  constexpr Derived FastCyclotomicInverse() const {
    // As the multiplicative subgroup is of order p² - 1, the
    // only non-trivial cyclotomic subgroup is of order p + 1
    // Therefore, for any element in the cyclotomic subgroup, we have that
    // |xᵖ⁺¹ = 1|. Recall that |xᵖ⁺¹| in a quadratic extension
    // field is equal to the norm in the base field, so we have that
    // |x * x.Conjugate() = 1|. By uniqueness of inverses, for this subgroup,
    // |x.Inverse() = x.Conjugate()|.
    return Conjugate();
  }

  constexpr Derived& FastCyclotomicInverseInPlace() {
    // NOTE(chokobole): CHECK(!IsZero()) is not a device code.
    // See https://github.com/kroma-network/tachyon/issues/76
    if (IsZero()) return *static_cast<Derived*>(this);

    return ConjugateInPlace();
  }

 protected:
  constexpr static void DoMul(const Derived& a, const Derived& b, Derived& c) {
    // clang-format off
    // (a.c0, a.c1) * (b.c0, b.c1)
    //   = (a.c0 + a.c1 * x) * (b.c0 + b.c1 * x)
    //   = a.c0 * b.c0 + (a.c0 * b.c1 + a.c1 * b.c0) * x + a.c1 * b.c1 * x²
    //   = a.c0 * b.c0 + a.c1 * b.c1 * x² + (a.c0 * b.c1 + a.c1 * b.c0) * x
    //   = a.c0 * b.c0 + a.c1 * b.c1 * q + (a.c0 * b.c1 + a.c1 * b.c0) * x
    //   = (a.c0 * b.c0 + a.c1 * b.c1 * q, a.c0 * b.c1 + a.c1 * b.c0)
    // Where q is Config::kNonResidue.
    // clang-format on
    if constexpr (ExtensionDegree() == 2) {
      BaseField c0;
      {
        BaseField lefts[] = {a.c0_, Config::MulByNonResidue(a.c1_)};
        BaseField rights[] = {b.c0_, b.c1_};
        c0 = BaseField::SumOfProductsSerial(lefts, rights);
      }
      {
        BaseField lefts[] = {a.c0_, a.c1_};
        BaseField rights[] = {b.c1_, b.c0_};
        c.c1_ = BaseField::SumOfProductsSerial(lefts, rights);
      }
      c.c0_ = std::move(c0);
    } else {
      // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
      // Karatsuba multiplication;
      // Guide to Pairing-based cryptography, Algorithm 5.16.
      // v0 = a.c0 * b.c0
      BaseField v0 = a.c0_ * b.c0_;
      // v1 = a.c1 * b.c1
      BaseField v1 = a.c1_ * b.c1_;

      // c.c1 = a.c0 + a.c1
      c.c1_ = a.c0_ + a.c1_;
      // c.c1 = (a.c0 + a.c1) * (b.c0 + b.c1)
      // c.c1 = a.c0 * b.c0 + a.c0 * b.c1 + a.c1 * b.c0 + a.c1 * b.c1
      c.c1_ *= (b.c0_ + b.c1_);
      // c.c1 = a.c0 * b.c1 + a.c1 * b.c0 + a.c1 * b.c1
      c.c1_ -= v0;
      // c.c1 = a.c0 * b.c1 + a.c1 * b.c0
      c.c1_ -= v1;
      // c.c0 = a.c0 * b.c0 + a.c1 * b.c1 * q
      c.c0_ = v0 + Config::MulByNonResidue(v1);
    }
  }

  constexpr static void DoSquareImpl(const Derived& a, Derived& b) {
    // (c0, c1)² = (c0 + c1 * x)²
    //            = c0² + 2 * c0 * c1 * x + c1² * x²
    //            = c0² + c1² * x² + 2 * c0 * c1 * x
    //            = c0² + c1² * q + 2 * c0 * c1 * x
    //            = (c0² + c1² * q, 2 * c0 * c1)
    // Where q is Config::kNonResidue.
    // When q = -1, we can re-use intermediate additions to improve performance.

    // v0 = c0 - c1
    BaseField v0 = a.c0_ - a.c1_;
    // v1 = c0 * c1
    BaseField v1 = a.c0_ * a.c1_;
    if constexpr (Config::kNonResidueIsMinusOne) {
      // When the non-residue is -1, we save 2 intermediate additions,
      // and use one fewer intermediate variable

      // v0 = (c0 - c1) * (c0 + c1)
      //    = c0² - c1²
      v0 *= (a.c0_ + a.c1_);

      // c0 = c0² - c1²
      b.c0_ = std::move(v0);
      // c1 = 2 * c0 * c1
      b.c1_ = v1.Double();
    } else {
      // v2 = c0 - q * c1
      BaseField v2 = a.c0_ - Config::MulByNonResidue(a.c1_);

      // v0 = (v0 * v2)
      // v0 = (c0 - c1) * (c0 - c1 * q)
      // v0 = c0² - c0 * c1 * q - c0 * c1 + c1² * q
      // v0 = c0² - (q + 1) * c0 * c1 + c1² * q
      // v0 = c0² + c1² * q - (q + 1) * c0 * c1
      v0 *= v2;

      // c0 = v0 + (q + 1) * c0 * c1
      // c0 = c0² + c1² * q - (q + 1) * c0 * c1 + (q + 1) * c0 * c1
      // c0 = c0² + c1² * q
      b.c0_ = v0 + v1;
      b.c0_ += Config::MulByNonResidue(v1);
      // c1 = 2 * c0 * c1
      b.c1_ = v1.Double();
    }
  }

  constexpr static void DoInverse(const Derived& a, Derived& b) {
    // NOTE(chokobole): CHECK(!IsZero()) is not a device code.
    // See https://github.com/kroma-network/tachyon/issues/76
    if (a.IsZero()) return;
    // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
    // Guide to Pairing-based Cryptography, Algorithm 5.19.
    // v1 = c1²
    BaseField v1 = a.c1_.Square();
    // v0 = c0² - q * v1
    BaseField v0 = a.c0_.Square();
    v0 -= Config::MulByNonResidue(v1);

    v1 = v0.Inverse();
    b.c0_ = a.c0_ * v1;
    b.c1_ = a.c1_ * v1;
    b.c1_.NegInPlace();
  }

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
      const ReadOnlyBuffer& buffer,
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
    return base::EstimateSize(quadratic_extension_field.c0(),
                              quadratic_extension_field.c1());
  }
};

template <typename Derived>
class RapidJsonValueConverter<
    Derived, std::enable_if_t<std::is_base_of_v<
                 math::QuadraticExtensionField<Derived>, Derived>>> {
 public:
  using BaseField = typename math::QuadraticExtensionField<Derived>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(
      const math::QuadraticExtensionField<Derived>& value,
      Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "c0", value.c0(), allocator);
    AddJsonElement(object, "c1", value.c1(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::QuadraticExtensionField<Derived>* value,
                 std::string* error) {
    BaseField c0;
    BaseField c1;
    if (!ParseJsonElement(json_value, "c0", &c0, error)) return false;
    if (!ParseJsonElement(json_value, "c1", &c1, error)) return false;
    *value =
        math::QuadraticExtensionField<Derived>(std::move(c0), std::move(c1));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_QUADRATIC_EXTENSION_FIELD_H_
