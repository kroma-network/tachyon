// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_CUBIC_EXTENSION_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_CUBIC_EXTENSION_FIELD_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "absl/types/span.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/json/json.h"
#include "tachyon/math/finite_fields/cyclotomic_multiplicative_subgroup.h"

namespace tachyon {
namespace math {

template <typename Derived>
class CubicExtensionField : public CyclotomicMultiplicativeSubgroup<Derived> {
 public:
  using Config = typename FiniteField<Derived>::Config;
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;

  constexpr CubicExtensionField() = default;
  constexpr CubicExtensionField(const BaseField& c0, const BaseField& c1,
                                const BaseField& c2)
      : c0_(c0), c1_(c1), c2_(c2) {}
  constexpr CubicExtensionField(BaseField&& c0, BaseField&& c1, BaseField&& c2)
      : c0_(std::move(c0)), c1_(std::move(c1)), c2_(std::move(c2)) {}

  constexpr static Derived Zero() {
    return {BaseField::Zero(), BaseField::Zero(), BaseField::Zero()};
  }

  constexpr static Derived One() {
    return {BaseField::One(), BaseField::Zero(), BaseField::Zero()};
  }

  static Derived Random() {
    return {BaseField::Random(), BaseField::Random(), BaseField::Random()};
  }

  static Derived FromBasePrimeFields(
      absl::Span<const BasePrimeField> prime_fields) {
    CHECK_EQ(prime_fields.size(), ExtensionDegree());
    constexpr size_t kBaseFieldDegree = BaseField::ExtensionDegree();
    if constexpr (kBaseFieldDegree == 1) {
      return Derived(prime_fields[0], prime_fields[1], prime_fields[2]);
    } else {
      BaseField c0 = BaseField::FromBasePrimeFields(
          prime_fields.subspan(0, kBaseFieldDegree));
      BaseField c1 = BaseField::FromBasePrimeFields(
          prime_fields.subspan(kBaseFieldDegree, kBaseFieldDegree));
      BaseField c2 = BaseField::FromBasePrimeFields(
          prime_fields.subspan(2 * kBaseFieldDegree, kBaseFieldDegree));
      return Derived(std::move(c0), std::move(c1), std::move(c2));
    }
  }

  constexpr bool IsZero() const {
    return c0_.IsZero() && c1_.IsZero() && c2_.IsZero();
  }

  constexpr bool IsOne() const {
    return c0_.IsOne() && c1_.IsZero() && c2_.IsZero();
  }

  constexpr static uint64_t ExtensionDegree() {
    return 3 * BaseField::ExtensionDegree();
  }

  // Calculate the norm of an element with respect to |BaseField|.
  // The norm maps an element |a| in the extension field Fqᵐ to an element
  // in the |BaseField| Fq. |a.Norm() = a * a^q * a^q²|
  constexpr BaseField Norm() const {
    // w.r.t to |BaseField|, we need the 0th, 1st & 2nd powers of q.
    // Since Frobenius coefficients on the towered extensions are
    // indexed w.r.t. to |BasePrimeField|, we need to calculate the correct
    // index.
    // NOTE(chokobole): This assumes that |BaseField::ExtensionDegree()|
    // never overflows even on a 32-bit machine.
    size_t index_multiplier = size_t{BaseField::ExtensionDegree()};
    Derived self_to_p = static_cast<const Derived&>(*this);
    self_to_p.FrobeniusMapInPlace(index_multiplier);
    Derived self_to_p2 = static_cast<const Derived&>(*this);
    self_to_p2.FrobeniusMapInPlace(2 * index_multiplier);
    self_to_p *= (self_to_p2 * static_cast<const Derived&>(*this));
    // NOTE(chokobole): The |CHECK()| below is not device code.
    // See https://github.com/kroma-network/tachyon/issues/76
    CHECK(self_to_p.c1().IsZero() && self_to_p.c2().IsZero());
    return self_to_p.c0();
  }

  constexpr Derived& FrobeniusMapInPlace(uint64_t exponent) {
    c0_.FrobeniusMapInPlace(exponent);
    c1_.FrobeniusMapInPlace(exponent);
    c2_.FrobeniusMapInPlace(exponent);
    c1_ *=
        Config::kFrobeniusCoeffs[exponent % Config::kDegreeOverBasePrimeField];
    c2_ *=
        Config::kFrobeniusCoeffs2[exponent % Config::kDegreeOverBasePrimeField];
    return *static_cast<Derived*>(this);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2)", c0_.ToString(), c1_.ToString(),
                            c2_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2)", c0_.ToHexString(pad_zero),
                            c1_.ToHexString(pad_zero),
                            c2_.ToHexString(pad_zero));
  }

  constexpr const BaseField& c0() const { return c0_; }
  constexpr const BaseField& c1() const { return c1_; }
  constexpr const BaseField& c2() const { return c2_; }

  constexpr bool operator==(const Derived& other) const {
    return c0_ == other.c0_ && c1_ == other.c1_ && c2_ == other.c2_;
  }

  constexpr bool operator!=(const Derived& other) const {
    return !operator==(other);
  }

  constexpr bool operator<(const Derived& other) const {
    if (c2_ == other.c2_) {
      if (c1_ == other.c1_) return c0_ < other.c0_;
      return c1_ < other.c1_;
    }
    return c2_ < other.c2_;
  }

  constexpr bool operator>(const Derived& other) const {
    if (c2_ == other.c2_) {
      if (c1_ == other.c1_) return c0_ > other.c0_;
      return c1_ > other.c1_;
    }
    return c2_ > other.c2_;
  }

  constexpr bool operator<=(const Derived& other) const {
    return !operator>(other);
  }

  constexpr bool operator>=(const Derived& other) const {
    return !operator<(other);
  }

  // AdditiveSemigroup methods
  constexpr Derived Add(const Derived& other) const {
    return {
        c0_ + other.c0_,
        c1_ + other.c1_,
        c2_ + other.c2_,
    };
  }

  constexpr Derived& AddInPlace(const Derived& other) {
    c0_ += other.c0_;
    c1_ += other.c1_;
    c2_ += other.c2_;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived DoubleImpl() const {
    return {
        c0_.Double(),
        c1_.Double(),
        c2_.Double(),
    };
  }

  constexpr Derived& DoubleImplInPlace() {
    c0_.DoubleInPlace();
    c1_.DoubleInPlace();
    c2_.DoubleInPlace();
    return *static_cast<Derived*>(this);
  }

  // AdditiveGroup methods
  constexpr Derived Sub(const Derived& other) const {
    return {
        c0_ - other.c0_,
        c1_ - other.c1_,
        c2_ - other.c2_,
    };
  }

  constexpr Derived& SubInPlace(const Derived& other) {
    c0_ -= other.c0_;
    c1_ -= other.c1_;
    c2_ -= other.c2_;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived Negate() const {
    return {
        -c0_,
        -c1_,
        -c2_,
    };
  }

  constexpr Derived& NegateInPlace() {
    c0_.NegateInPlace();
    c1_.NegateInPlace();
    c2_.NegateInPlace();
    return *static_cast<Derived*>(this);
  }

  // MultiplicativeSemigroup methods
  constexpr Derived Mul(const Derived& other) const {
    Derived ret{};
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
        c2_ * element,
    };
  }

  constexpr Derived& MulInPlace(const BaseField& element) {
    c0_ *= element;
    c1_ *= element;
    c2_ *= element;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived SquareImpl() const {
    Derived ret{};
    DoSquareImpl(*static_cast<const Derived*>(this), ret);
    return ret;
  }

  constexpr Derived& SquareImplInPlace() {
    DoSquareImpl(*static_cast<const Derived*>(this),
                 *static_cast<Derived*>(this));
    return *static_cast<Derived*>(this);
  }

  // MultiplicativeGroup methods
  constexpr std::optional<Derived> Inverse() const {
    Derived ret{};
    if (LIKELY(DoInverse(*static_cast<const Derived*>(this), ret))) return ret;
    LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
    return std::nullopt;
  }

  [[nodiscard]] constexpr std::optional<Derived*> InverseInPlace() {
    if (LIKELY(DoInverse(*static_cast<const Derived*>(this),
                         *static_cast<Derived*>(this)))) {
      return static_cast<Derived*>(this);
    }
    LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
    return std::nullopt;
  }

 protected:
  template <typename Config, typename SFINAE>
  friend class Fp6;
  template <typename Config>
  friend class Fp12;

  constexpr static void DoMul(const Derived& a, const Derived& b, Derived& c) {
    // clang-format off
    // (a.c0, a.c1, a.c2) * (b.c0, b.c1, b.c2)
    //   = (a.c0 + a.c1 * x + a.c2 * x²) * (b.c0 + b.c1 * x + b.c2 * x²)
    //   = a.c0 * b.c0 + (a.c0 * b.c1 + a.c1 * b.c0) * x + (a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0) * x² +
    //     (a.c1 * b.c2 + a.c2 * b.c1) * x³ + a.c2 * b.c2 * x⁴
    //   = a.c0 * b.c0 + (a.c1 * b.c2 + a.c2 * b.c1) * x³ +
    //     (a.c0 * b.c1 + a.c1 * b.c0) * x + a.c2 * b.c2 * x⁴ +
    //     (a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0) * x²
    //   = a.c0 * b.c0 + (a.c1 * b.c2 + a.c2 * b.c1) * q +
    //     (a.c0 * b.c1 + a.c1 * b.c0) * x + a.c2 * b.c2 * q * x +
    //     (a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0) * x²
    //   = (a.c0 * b.c0 + (a.c1 * b.c2 + a.c2 * b.c1) * q,
    //     a.c0 * b.c1 + a.c1 * b.c0 + a.c2 * b.c2 * q,
    //     a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0)
    // where q is |Config::kNonResidue|.

    // See https://eprint.iacr.org/2006/471.pdf
    // Devegili OhEig Scott Dahab --- Multiplication and Squaring on AbstractPairing-Friendly Fields.pdf; Section 4 (Karatsuba)
    // clang-format on

    BaseField v0 = a.c0_ * b.c0_;
    BaseField v1 = a.c1_ * b.c1_;
    BaseField v2 = a.c2_ * b.c2_;

    // x = a.c0 * b.c1 + a.c1 * b.c0
    BaseField x = (a.c0_ + a.c1_) * (b.c0_ + b.c1_) - v0 - v1;
    // y = a.c0 * b.c2 + a.c2 * b.c0
    BaseField y = (a.c0_ + a.c2_) * (b.c0_ + b.c2_) - v0 - v2;
    // z = a.c1 * b.c2 + a.c2 * b.c1
    BaseField z = (a.c1_ + a.c2_) * (b.c1_ + b.c2_) - v1 - v2;

    // c.c0 = a.c0 * b.c0 + (a.c1 * b.c2 + a.c2 * b.c1) * q
    c.c0_ = v0 + Config::MulByNonResidue(z);
    // c.c1 = a.c0 * b.c1 + a.c1 * b.c0 + a.c2 * b.c2 * q
    c.c1_ = x + Config::MulByNonResidue(v2);
    // c.c2 = a.c0 * b.c2 + a.c2 * b.c0 + a.c1 * b.c1
    c.c2_ = y + v1;
  }

  constexpr static void DoSquareImpl(const Derived& a, Derived& b) {
    // clang-format off
    // (c0, c1, c2)²
    //   = (c0 + c1 * x + c2 * x²)²
    //   = c0² + 2 * c0 * c1 * x + (c1² + 2 * c0 * c2) * x² + 2 * c1 * c2 * x³ + c2² * x⁴
    //   = c0² + 2 * c0 * c1 * x + 2 * c1 * c2 * x³ + (c1² + 2 * c0 * c2) * x² + c2² * x⁴
    //   = c0² + 2 * c1 * c2 * x³ + 2 * c0 * c1 * x + c2² * x⁴ + (c1² + 2 * c0 * c2) * x²
    //   = c0² + 2 * c1 * c2 * q + (2 * c0 * c1  + c2² * q) * x + (c1² + 2 * c0 * c2) * x²
    //   = (c0² + 2 * c1 * c2 * q, 2 * c0 * c1  + c2² * q, c1² + 2 * c0 * c2)
    // where q is |Config::kNonResidue|.

    // See https://eprint.iacr.org/2006/471.pdf
    // Devegili OhEig Scott Dahab --- Multiplication and Squaring on AbstractPairing-Friendly Fields.pdf; Section 4 (CH-SQR2)
    // clang-format on

    // s0 = c0²
    BaseField s0 = a.c0_.Square();
    // s1 = 2 * c0 * c1
    BaseField s1 = (a.c0_ * a.c1_).Double();
    // s2 = (c0 - c1 + c2)²
    //    = c0² + c1² + c2² - 2 * c0 * c1 - 2 c1 * c2 + 2 * c0 * c2
    BaseField s2 = (a.c0_ - a.c1_ + a.c2_).Square();
    // s3 = 2 * c1 * c2
    BaseField s3 = (a.c1_ * a.c2_).Double();
    // s4 = c2²
    BaseField s4 = a.c2_.Square();

    // c0 = c0² + 2 * c1 * c2 * q
    b.c0_ = s0 + Config::MulByNonResidue(s3);
    // c1 = 2 * c0 * c1 + c2² * q
    b.c1_ = s1 + Config::MulByNonResidue(s4);
    // c2 = 2 * c0 * c1 +
    //      c0² + c1² + c2² - 2 * c0 * c1 - 2 c1 * c2 + 2 * c0 * c2 +
    //      2 * c1 * c2 -
    //      c0² -
    //      c2²
    //    = c1² + 2 * c0 * c2
    b.c2_ = s1 + s2 + s3 - s0 - s4;
  }

  [[nodiscard]] constexpr static bool DoInverse(const Derived& a, Derived& b) {
    if (UNLIKELY(a.IsZero())) {
      LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
      return false;
    }
    // clang-format off
    // See https://eprint.iacr.org/2010/354.pdf
    // From "High-Speed Software Implementation of the Optimal Ate AbstractPairing over Barreto-Naehrig Curves"; Algorithm 17
    // clang-format on

    // t0 = c0²
    BaseField t0 = a.c0_.Square();
    // t1 = c1²
    BaseField t1 = a.c1_.Square();
    // t2 = c2²
    BaseField t2 = a.c2_.Square();
    // t3 = c0 * c1
    BaseField t3 = a.c0_ * a.c1_;
    // t4 = c0 * c2
    BaseField t4 = a.c0_ * a.c2_;
    // t5 = c1 * c2
    BaseField t5 = a.c1_ * a.c2_;

    // s0 = t0 - t5 * q
    //    = t0 - c1 * c2 * q
    BaseField s0 = t0 - Config::MulByNonResidue(t5);
    // s1 = t2 * q - t3
    //    = c2² * q - c0 * c1
    BaseField s1 = Config::MulByNonResidue(t2) - t3;
    // See
    // https://github.com/arkworks-rs/algebra/blob/c92be0e8815875460e736086a6b02fed9e4273ff/ff/src/fields/models/cubic_extension.rs#L315
    // s2 = t1 - t4
    //    = c1² - c0 * c2
    BaseField s2 = t1 - t4;

    // a1 = c2 * s1
    //    = c2 * (c2² * q - c0 * c1)
    //    = c2³ * q - c0 * c1 * c2
    BaseField a1 = a.c2_ * s1;
    // a2 = c1 * s2
    //    = c1 * (c1² - c0 * c2)
    //    = c1³ - c0 * c1 * c2
    BaseField a2 = a.c1_ * s2;
    // a3 = c1³ + c2³ * q - 2 * c0 * c1 * c2
    BaseField a3 = Config::MulByNonResidue(a1 + a2);
    // t6 = 1 / (c0 * s0 + a3)
    //    = 1 / (c0 * (t0 - t5 * q) + c1³ + c2³ * q - 2 * c0 * c1 * c2)
    //    = 1 / (c0 * (c0² - c1 * c2 * q) + c1³ + c2³ * q - 2 * c0 * c1 * c2)
    //    = 1 / (c0³ + c1³ + c2³ - (2 + q) * c0 * c1 * c2)
    const std::optional<BaseField> t6_opt = (a.c0_ * s0 + a3).Inverse();
    if (UNLIKELY(!t6_opt)) return false;
    BaseField t6 = std::move(*t6_opt);

    // c0 = s0 * t6
    // c0 = (t0 - c1 * c2 * q) / (c0³ + c1³ + c2³ - (2 + q) * c0 * c1 * c2)
    //    = (c0² - c1 * c2 * q) / (c0³ + c1³ + c2³ - (2 + q) * c0 * c1 * c2)
    b.c0_ = s0 * t6;
    // c1 = s1 * t6
    //    = (c2² * q - c0 * c1) / (c0³ + c1³ + c2³ - (2 + q) * c0 * c1 * c2)
    b.c1_ = s1 * t6;
    // c2 = s1 * t6
    //    = (c1² - c0 * c2) / (c0³ + c1³ + c2³ - (2 + q) * c0 * c1 * c2)
    b.c2_ = s2 * t6;
    return true;
  }

  // c = c0_ + c1_ * X + c2_ * X²
  BaseField c0_;
  BaseField c1_;
  BaseField c2_;
};

template <
    typename BaseField, typename Derived,
    std::enable_if_t<std::is_same_v<BaseField, typename Derived::BaseField>>* =
        nullptr>
Derived operator*(const BaseField& element,
                  const CubicExtensionField<Derived>& f) {
  return static_cast<const Derived&>(f) * element;
}

}  // namespace math

namespace base {

template <typename Derived>
class Copyable<Derived, std::enable_if_t<std::is_base_of_v<
                            math::CubicExtensionField<Derived>, Derived>>> {
 public:
  static bool WriteTo(
      const math::CubicExtensionField<Derived>& cubic_extension_field,
      Buffer* buffer) {
    return buffer->WriteMany(cubic_extension_field.c0(),
                             cubic_extension_field.c1(),
                             cubic_extension_field.c2());
  }

  static bool ReadFrom(
      const ReadOnlyBuffer& buffer,
      math::CubicExtensionField<Derived>* cubic_extension_field) {
    typename Derived::BaseField c0;
    typename Derived::BaseField c1;
    typename Derived::BaseField c2;
    if (!buffer.ReadMany(&c0, &c1, &c2)) return false;

    *cubic_extension_field = math::CubicExtensionField<Derived>(
        std::move(c0), std::move(c1), std::move(c2));
    return true;
  }

  static size_t EstimateSize(
      const math::CubicExtensionField<Derived>& cubic_extension_field) {
    return base::EstimateSize(cubic_extension_field.c0(),
                              cubic_extension_field.c1(),
                              cubic_extension_field.c2());
  }
};

template <typename Derived>
class RapidJsonValueConverter<
    Derived, std::enable_if_t<std::is_base_of_v<
                 math::CubicExtensionField<Derived>, Derived>>> {
 public:
  using BaseField = typename math::CubicExtensionField<Derived>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(const math::CubicExtensionField<Derived>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "c0", value.c0(), allocator);
    AddJsonElement(object, "c1", value.c1(), allocator);
    AddJsonElement(object, "c2", value.c2(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::CubicExtensionField<Derived>* value,
                 std::string* error) {
    BaseField c0;
    BaseField c1;
    BaseField c2;
    if (!ParseJsonElement(json_value, "c0", &c0, error)) return false;
    if (!ParseJsonElement(json_value, "c1", &c1, error)) return false;
    if (!ParseJsonElement(json_value, "c2", &c2, error)) return false;
    *value = math::CubicExtensionField<Derived>(std::move(c0), std::move(c1),
                                                std::move(c2));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_CUBIC_EXTENSION_FIELD_H_
