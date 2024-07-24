#ifndef TACHYON_MATH_FINITE_FIELDS_QUARTIC_EXTENSION_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_QUARTIC_EXTENSION_FIELD_H_

#include <array>
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
class QuarticExtensionField : public CyclotomicMultiplicativeSubgroup<Derived> {
 public:
  using Config = typename FiniteField<Derived>::Config;
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;

  constexpr QuarticExtensionField() = default;
  // NOTE(chokobole): This is needed by Eigen matrix.
  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  constexpr explicit QuarticExtensionField(T value) : c0_(value) {}
  constexpr explicit QuarticExtensionField(const BaseField& c0)
      : c0_(c0),
        c1_(BaseField::Zero()),
        c2_(BaseField::Zero()),
        c3_(BaseField::Zero()) {}
  constexpr explicit QuarticExtensionField(BaseField&& c0)
      : c0_(std::move(c0)),
        c1_(BaseField::Zero()),
        c2_(BaseField::Zero()),
        c3_(BaseField::Zero()) {}
  constexpr QuarticExtensionField(const BaseField& c0, const BaseField& c1,
                                  const BaseField& c2, const BaseField& c3)
      : c0_(c0), c1_(c1), c2_(c2), c3_(c3) {}
  constexpr QuarticExtensionField(BaseField&& c0, BaseField&& c1,
                                  BaseField&& c2, BaseField&& c3)
      : c0_(std::move(c0)),
        c1_(std::move(c1)),
        c2_(std::move(c2)),
        c3_(std::move(c3)) {}

  constexpr static Derived Zero() {
    return {BaseField::Zero(), BaseField::Zero(), BaseField::Zero(),
            BaseField::Zero()};
  }

  constexpr static Derived One() {
    return {BaseField::One(), BaseField::Zero(), BaseField::Zero(),
            BaseField::Zero()};
  }

  constexpr static Derived MinusOne() {
    return {BaseField::MinusOne(), BaseField::Zero(), BaseField::Zero(),
            BaseField::Zero()};
  }

  static Derived Random() {
    return {BaseField::Random(), BaseField::Random(), BaseField::Random(),
            BaseField::Random()};
  }

  // TODO(chokobole): Should be generalized for packed extension field.
  static Derived FromBasePrimeFields(
      absl::Span<const BasePrimeField> prime_fields) {
    CHECK_EQ(prime_fields.size(), ExtensionDegree());
    constexpr size_t kBaseFieldDegree = BaseField::ExtensionDegree();
    if constexpr (kBaseFieldDegree == 1) {
      return Derived(prime_fields[0], prime_fields[1], prime_fields[2],
                     prime_fields[3]);
    } else {
      BaseField c0 = BaseField::FromBasePrimeFields(
          prime_fields.subspan(0, kBaseFieldDegree));
      prime_fields.remove_prefix(kBaseFieldDegree);
      BaseField c1 = BaseField::FromBasePrimeFields(
          prime_fields.subspan(0, kBaseFieldDegree));
      prime_fields.remove_prefix(kBaseFieldDegree);
      BaseField c2 = BaseField::FromBasePrimeFields(
          prime_fields.subspan(0, kBaseFieldDegree));
      prime_fields.remove_prefix(kBaseFieldDegree);
      BaseField c3 = BaseField::FromBasePrimeFields(
          prime_fields.subspan(kBaseFieldDegree));
      return Derived(std::move(c0), std::move(c1), std::move(c2),
                     std::move(c3));
    }
  }

  constexpr std::array<BaseField, 4> ToBaseFields() const {
    return {c0_, c1_, c2_, c3_};
  }

  static void Init() {
    kInv2 = *BaseField(2).Inverse();
    kInv3 = *BaseField(3).Inverse();
    kInv4 = *BaseField(4).Inverse();
    kInv6 = *BaseField(6).Inverse();
    kInv12 = *BaseField(12).Inverse();
    kInv20 = *BaseField(20).Inverse();
    kInv24 = *BaseField(24).Inverse();
    kInv30 = *BaseField(30).Inverse();
    kInv120 = *BaseField(120).Inverse();
    kNeg5 = -BaseField(5);
    kNegInv2 = -kInv2;
    kNegInv3 = -kInv3;
    kNegInv4 = -kInv4;
    kNegInv6 = -kInv6;
    kNegInv12 = -kInv12;
    kNegInv24 = -kInv24;
    kNegInv120 = -kInv120;
  }

  constexpr bool IsZero() const {
    return c0_.IsZero() && c1_.IsZero() && c2_.IsZero() && c3_.IsZero();
  }

  constexpr bool IsOne() const {
    return c0_.IsOne() && c1_.IsZero() && c2_.IsZero() && c3_.IsZero();
  }

  constexpr bool IsMinusOne() const {
    return c0_.IsMinusOne() && c1_.IsZero() && c2_.IsZero() && c3_.IsZero();
  }

  constexpr static uint32_t ExtensionDegree() {
    return 4 * BaseField::ExtensionDegree();
  }

  // Calculate the norm of an element with respect to |BaseField|.
  // The norm maps an element |a| in the extension field Fqᵐ to an element
  // in the |BaseField| Fq. |a.Norm() = a * a^q * a^q² * a^q³|
  constexpr BaseField Norm() const {
    // w.r.t to |BaseField|, we need the 0th, 1st, 2nd & 3rd powers of q.
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
    Derived self_to_p3 = static_cast<const Derived&>(*this);
    self_to_p3.FrobeniusMapInPlace(3 * index_multiplier);
    self_to_p *= (self_to_p2 * self_to_p3 * static_cast<const Derived&>(*this));
    // NOTE(chokobole): The |CHECK()| below is not device code.
    // See https://github.com/kroma-network/tachyon/issues/76
    CHECK(self_to_p.c1().IsZero() && self_to_p.c2().IsZero() &&
          self_to_p.c3().IsZero());
    return self_to_p.c0();
  }

  constexpr Derived& FrobeniusMapInPlace(uint32_t exponent) {
    c0_.FrobeniusMapInPlace(exponent);
    c1_.FrobeniusMapInPlace(exponent);
    c2_.FrobeniusMapInPlace(exponent);
    c3_.FrobeniusMapInPlace(exponent);
    c1_ *=
        Config::kFrobeniusCoeffs[exponent % Config::kDegreeOverBasePrimeField];
    c2_ *=
        Config::kFrobeniusCoeffs2[exponent % Config::kDegreeOverBasePrimeField];
    c3_ *=
        Config::kFrobeniusCoeffs3[exponent % Config::kDegreeOverBasePrimeField];
    return *static_cast<Derived*>(this);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2, $3)", c0_.ToString(), c1_.ToString(),
                            c2_.ToString(), c3_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2, $3)", c0_.ToHexString(pad_zero),
                            c1_.ToHexString(pad_zero),
                            c2_.ToHexString(pad_zero),
                            c3_.ToHexString(pad_zero));
  }

  constexpr const BaseField& c0() const { return c0_; }
  constexpr const BaseField& c1() const { return c1_; }
  constexpr const BaseField& c2() const { return c2_; }
  constexpr const BaseField& c3() const { return c3_; }

  constexpr const BaseField& operator[](size_t index) const {
    switch (index) {
      case 0:
        return c0_;
      case 1:
        return c1_;
      case 2:
        return c2_;
      case 3:
        return c3_;
    }
    NOTREACHED();
    return c0_;
  }

  constexpr BaseField& operator[](size_t index) {
    return const_cast<BaseField&>(std::as_const(*this).operator[](index));
  }

  constexpr bool operator==(const Derived& other) const {
    return c0_ == other.c0_ && c1_ == other.c1_ && c2_ == other.c2_ &&
           c3_ == other.c3_;
  }

  constexpr bool operator!=(const Derived& other) const {
    return !operator==(other);
  }

  constexpr bool operator<(const Derived& other) const {
    if (c3_ == other.c3_) {
      if (c2_ == other.c2_) {
        if (c1_ == other.c1_) return c0_ < other.c0_;
        return c1_ < other.c1_;
      }
      return c2_ < other.c2_;
    }
    return c3_ < other.c3_;
  }

  constexpr bool operator>(const Derived& other) const {
    if (c3_ == other.c3_) {
      if (c2_ == other.c2_) {
        if (c1_ == other.c1_) return c0_ > other.c0_;
        return c1_ > other.c1_;
      }
      return c2_ > other.c2_;
    }
    return c3_ > other.c3_;
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
        c3_ + other.c3_,
    };
  }

  constexpr Derived& AddInPlace(const Derived& other) {
    c0_ += other.c0_;
    c1_ += other.c1_;
    c2_ += other.c2_;
    c3_ += other.c3_;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived DoubleImpl() const {
    return {
        c0_.Double(),
        c1_.Double(),
        c2_.Double(),
        c3_.Double(),
    };
  }

  constexpr Derived& DoubleImplInPlace() {
    c0_.DoubleInPlace();
    c1_.DoubleInPlace();
    c2_.DoubleInPlace();
    c3_.DoubleInPlace();
    return *static_cast<Derived*>(this);
  }

  // AdditiveGroup methods
  constexpr Derived Sub(const Derived& other) const {
    return {
        c0_ - other.c0_,
        c1_ - other.c1_,
        c2_ - other.c2_,
        c3_ - other.c3_,
    };
  }

  constexpr Derived& SubInPlace(const Derived& other) {
    c0_ -= other.c0_;
    c1_ -= other.c1_;
    c2_ -= other.c2_;
    c3_ -= other.c3_;
    return *static_cast<Derived*>(this);
  }

  constexpr Derived Negate() const {
    return {
        -c0_,
        -c1_,
        -c2_,
        -c3_,
    };
  }

  constexpr Derived& NegateInPlace() {
    c0_.NegateInPlace();
    c1_.NegateInPlace();
    c2_.NegateInPlace();
    c3_.NegateInPlace();
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
        c3_ * element,
    };
  }

  constexpr Derived& MulInPlace(const BaseField& element) {
    c0_ *= element;
    c1_ *= element;
    c2_ *= element;
    c3_ *= element;
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
    if (LIKELY(DoInverse(*static_cast<const Derived*>(this), ret))) {
      return ret;
    }
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
  constexpr static void DoMul(const Derived& a, const Derived& b, Derived& c) {
    // clang-format off
    // (a.c0, a.c1, a.c2, a.c3) * (b.c0, b.c1, b.c2, b.c3)
    //   = (a.c0 + a.c1 * x + a.c2 * x² + a.c3 * x³) * (b.c0 + b.c1 * x + b.c2 * x² + b.c3 * x³)
    //   = a.c0 * b.c0 + (a.c0 * b.c1 + a.c1 * b.c0) * x + (a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0) * x² +
    //     (a.c0 * b.c3 + a.c1 * b.c2 + a.c2 * b.c1 + a.c3 * b.c0) * x³ + (a.c1 * b.c3 + a.c2 * b.c2 + a.c3 * b.c1) * x⁴ +
    //     (a.c2 * b.c3 + a.c3 * b.c2) * x⁵ + a.c3 * b.c3 * x⁶
    //   = a.c0 * b.c0 + (a.c1 * b.c3 + a.c2 * b.c2 + a.c3 * b.c1) * x⁴ +
    //     (a.c0 * b.c1 + a.c1 * b.c0) * x + (a.c2 * b.c3 + a.c3 * b.c2) * x⁵ +
    //     (a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0) * x² + a.c3 * b.c3 * x⁶ +
    //     (a.c0 * b.c3 + a.c1 * b.c2 + a.c2 * b.c1 + a.c3 * b.c0) * x³
    //   = a.c0 * b.c0 + (a.c1 * b.c3 + a.c2 * b.c2 + a.c3 * b.c1) * q +
    //     (a.c0 * b.c1 + a.c1 * b.c0) * x + (a.c2 * b.c3 + a.c3 * b.c2) * q * x +
    //     (a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0) * x² + a.c3 * b.c3 * q * x² +
    //     (a.c0 * b.c3 + a.c1 * b.c2 + a.c2 * b.c1 + a.c3 * b.c0) * x³
    //   = (a.c0 * b.c0 + (a.c1 * b.c3 + a.c2 * b.c2 + a.c3 * b.c1) * q,
    //      a.c0 * b.c1 + a.c1 * b.c0 + (a.c2 * b.c3 + a.c3 * b.c2) * q,
    //      a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0 + a.c3 * b.c3 * q,
    //      a.c0 * b.c3 + a.c1 * b.c2 + a.c2 * b.c1 + a.c3 * b.c0)
    // where q is |Config::kNonResidue|.

    // See https://eprint.iacr.org/2006/471.pdf
    // Devegili OhEig Scott Dahab --- Multiplication and Squaring on AbstractPairing-Friendly Fields.pdf; Section 5.2
    // clang-format on

    // h1 = 2 * a.c1
    BaseField h1 = a.c1_.Double();
    // h2 = 4 * a.c2
    BaseField h2 = a.c2_.Double();
    h2.DoubleInPlace();
    // h3 = 8 * a.c3
    BaseField h3 = a.c3_.Double();
    h3.DoubleInPlace().DoubleInPlace();
    // h4 = 2 * b.c1
    BaseField h4 = b.c1_.Double();
    // h5 = 4 * b.c2
    BaseField h5 = b.c2_.Double();
    h5.DoubleInPlace();
    // h6 = 8 * b.c3
    BaseField h6 = b.c3_.Double();
    h6.DoubleInPlace().DoubleInPlace();

    // v0 = a.c0 * b.c0
    BaseField v0 = a.c0_ * b.c0_;
    // v1 = (a.c0 + a.c1 + a.c2 + a.c3) * (b.c0 + b.c1 + b.c2 + b.c3)
    BaseField v1 =
        (a.c0_ + a.c1_ + a.c2_ + a.c3_) * (b.c0_ + b.c1_ + b.c2_ + b.c3_);
    // v2 = (a.c0 - a.c1 + a.c2 - a.c3) * (b.c0 - b.c1 + b.c2 - b.c3)
    BaseField v2 =
        (a.c0_ - a.c1_ + a.c2_ - a.c3_) * (b.c0_ - b.c1_ + b.c2_ - b.c3_);
    // v3 = (a.c0 + 2 * a.c1 + 4 * a.c2 + 8 * a.c3) *
    //      (b.c0 + 2 * b.c1 + 4 * b.c2 + 8 * b.c3)
    BaseField v3 = (a.c0_ + h1 + h2 + h3) * (b.c0_ + h4 + h5 + h6);
    // v4 = (a.c0 - 2 * a.c1 + 4 * a.c2 - 8 * a.c3) *
    //      (b.c0 - 2 * b.c1 + 4 * b.c2 - 8 * b.c3)
    BaseField v4 = (a.c0_ - h1 + h2 - h3) * (b.c0_ - h4 + h5 - h6);
    // h1 = 3 * a.c1
    h1 += a.c1_;
    // h2 = 9 * a.c2
    h2.DoubleInPlace().AddInPlace(a.c2_);
    // h3 = 27 * a.c3
    h3 += a.c3_;
    h3 += h3.Double();
    // h4 = 3 * b.c1
    h4 += b.c1_;
    // h5 = 9 * b.c2
    h5.DoubleInPlace().AddInPlace(b.c2_);
    // h6 = 27 * b.c3
    h6 += b.c3_;
    h6 += h6.Double();
    // v5 = (a.c0 + 3 * a.c1 + 9 * a.c2 + 27 * a.c3) *
    //      (b.c0 + 3 * b.c1 + 9 * b.c2 + 27 * b.c3)
    BaseField v5 = (a.c0_ + h1 + h2 + h3) * (b.c0_ + h4 + h5 + h6);
    // v6 = a.c3 * b.c3
    BaseField v6 = a.c3_ * b.c3_;

    // v0_5 = 5 * v0
    BaseField v0_5 = v0.Double();
    v0_5.DoubleInPlace().AddInPlace(v0);
    // v6_3 = 3 * v6
    BaseField v6_3 = v6.Double();
    v6_3 += v6;

    // clang-format off
    // c.c0 = v0 +
    //        q * ((1 / 4) * v0 - (1 / 6) * (v1 + v2) + (1 / 24) * (v3 + v4) - 5 * v6)
    c.c0_ = v0 +
            Config::MulByNonResidue(kInv4 * v0 + kNegInv6 * (v1 + v2) + kInv24 * (v3 + v4) + kNeg5 * v6);
    // c.c1 = -(1 / 3) * v0 + v1 - (1 / 2) * v2 - (1 / 4) * v3 + (1 / 20) * v4 + (1 / 30) * v5 - 12 * v6 +
    //        q * (-(1 / 12) * (v0 - v1) + (1 / 24) * (v2 - v3) - (1 / 120) * (v4 - v5) - 3 * v6)
    c.c1_ = kNegInv3 * v0 + v1 + kNegInv2 * v2 + kNegInv4 * v3 + kInv20 * v4 + kInv30 * v5 - v6_3.Double().Double() +
            Config::MulByNonResidue(kNegInv12 * (v0 - v1) + kInv24 * (v2 - v3) + kNegInv120 * (v4 - v5) - v6_3);
    // c.c2 = -(5 / 4) * v0 + (2 / 3) * (v1 + v2) - (1 / 24) * (v3 + v4) + 4 * v6 +
    //        q * v6
    c.c2_ = kNegInv4 * v0_5 + kInv3 * (v1 + v2).Double() + kNegInv24 * (v3 + v4) + v6.Double().Double() +
            Config::MulByNonResidue(v6);
    // c.c3 = (1 / 12) * (5 * v0 - 7 * v1) - (1 / 24) * (v2 - 7 * v3 + v4 + v5) + 15 * v6
    c.c3_ = kInv12 * (v0_5 - v1.Double().Double().Double() + v1) + kNegInv24 * (v2 - v3.Double().Double().Double() + v3 + v4 + v5) + v6_3.Double().Double() + v6_3;
    // clang-format on
  }

  constexpr static void DoSquareImpl(const Derived& a, Derived& b) {
    // clang-format off
    // (c0, c1, c2, c3)²
    //   = (c0 + c1 * x + c2 * x² + c3 * x³)²
    //   = c0² + 2 * c0 * c1 * x + (c1² + 2 * c0 * c2) * x² + 2 * (c0 * c3 + c1 * c2) * x³ + (c2² + 2 * c1 * c3) * x⁴ + 2 * c2 * c3 * x⁵ + c3 * x⁶
    //   = c0² + (c2² + 2 * c1 * c3) * x⁴ + 2 * c0 * c1 * x + 2 * c2 * c3 * x⁵ + (c1² + 2 * c0 * c2) * x² + c3 * x⁶ + 2 * (c0 * c3 + c1 * c2) * x³
    //   = c0² + (c2² + 2 * c1 * c3) * q + 2 * (c0 * c1 + c2 * c3 * q) * x + (c1² + 2 * c0 * c2 + c3 * q) * x² + 2 * (c0 * c3 + c1 * c2) * x³
    //   = (c0² + (c2² + 2 * c1 * c3) * q, 2 * (c0 * c1 + c2 * c3 * q), c1² + 2 * c0 * c2 + c3 * q, 2 * (c0 * c3 + c1 * c2))
    // where q is |Config::kNonResidue|.

    // See https://eprint.iacr.org/2006/471.pdf
    // Devegili OhEig Scott Dahab --- Multiplication and Squaring on AbstractPairing-Friendly Fields.pdf; Section 5
    // clang-format on

    // h1 = 2 * c1
    BaseField h1 = a.c1_.Double();
    // h2 = 4 * c2
    BaseField h2 = a.c2_.Double();
    h2.DoubleInPlace();
    // h3 = 8 * c3
    BaseField h3 = a.c3_.Double();
    h3.DoubleInPlace().DoubleInPlace();

    // v0 = c0²
    BaseField v0 = a.c0_.Square();
    // v1 = (c0 + c1 + c2 + c3)²
    BaseField v1 = (a.c0_ + a.c1_ + a.c2_ + a.c3_).Square();
    // v2 = (c0 - c1 + c2 - c3)²
    BaseField v2 = (a.c0_ - a.c1_ + a.c2_ - a.c3_).Square();
    // v3 = (c0 + 2 * c1 + 4 * c2 + 8 * c3)²
    BaseField v3 = (a.c0_ + h1 + h2 + h3).Square();
    // v4 = (c0 - 2 * c1 + 4 * c2 - 8 * c3)²
    BaseField v4 = (a.c0_ - h1 + h2 - h3).Square();
    // h1 = 3 * c1
    h1 += a.c1_;
    // h2 = 9 * c2
    h2.DoubleInPlace().AddInPlace(a.c2_);
    // h3 = 27 * c3
    h3 += a.c3_;
    h3 += h3.Double();
    // v5 = (c0 + 3 * c1 + 9 * c2 + 27 * c3)²
    BaseField v5 = (a.c0_ + h1 + h2 + h3).Square();
    // v6 = c3²
    BaseField v6 = a.c3_.Square();

    // v0_5 = 5 * v0
    BaseField v0_5 = v0.Double();
    v0_5.DoubleInPlace().AddInPlace(v0);
    // v6_3 = 3 * v6
    BaseField v6_3 = v6.Double();
    v6_3 += v6;

    // clang-format off
    // b.c0 = v0 +
    //        q * ((1 / 4) * v0 - (1 / 6) * (v1 + v2) + (1 / 24) * (v3 + v4) - 5 * v6)
    b.c0_ = v0 +
            Config::MulByNonResidue(kInv4 * v0 + kNegInv6 * (v1 + v2) + kInv24 * (v3 + v4) + kNeg5 * v6);
    // b.c1 = -(1 / 3) * v0 + v1 - (1 / 2) * v2 - (1 / 4) * v3 + (1 / 20) * v4 + (1 / 30) * v5 - 12 * v6 +
    //        q * (-(1 / 12) * (v0 - v1) + (1 / 24) * (v2 - v3) - (1 / 120) * (v4 - v5) - 3 * v6)
    b.c1_ = kNegInv3 * v0 + v1 + kNegInv2 * v2 + kNegInv4 * v3 + kInv20 * v4 + kInv30 * v5 - v6_3.Double().Double() +
            Config::MulByNonResidue(kNegInv12 * (v0 - v1) + kInv24 * (v2 - v3) + kNegInv120 * (v4 - v5) - v6_3);
    // b.c2 = -(5 / 4) * v0 + (2 / 3) * (v1 + v2) - (1 / 24) * (v3 + v4) + 4 * v6 +
    //        q * v6
    b.c2_ = kNegInv4 * v0_5 + kInv3 * (v1 + v2).Double() + kNegInv24 * (v3 + v4) + v6.Double().Double() +
            Config::MulByNonResidue(v6);
    // b.c3 = (1 / 12) * (5 * v0 - 7 * v1) - (1 / 24) * (v2 - 7 * v3 + v4 + v5) + 15 * v6
    b.c3_ = kInv12 * (v0_5 - v1.Double().Double().Double() + v1) + kNegInv24 * (v2 - v3.Double().Double().Double() + v3 + v4 + v5) + v6_3.Double().Double() + v6_3;
    // clang-format on
  }

  [[nodiscard]] constexpr static bool DoInverse(const Derived& a, Derived& b) {
    if (UNLIKELY(a.IsZero())) {
      LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
      return false;
    }

    // See Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve
    // Cryptography.
    // Compute aʳ⁻¹, where r = (p⁴ - 1) / (p - 1) = p³ + p² + p + 1
    size_t index_multiplier = size_t{BaseField::ExtensionDegree()};
    // f = a^{p³ + p² + p}
    Derived a_to_r_minus_1 = a;
    a_to_r_minus_1.FrobeniusMapInPlace(index_multiplier);
    Derived a_to_p2 = a;
    a_to_p2.FrobeniusMapInPlace(2 * index_multiplier);
    Derived a_to_p3 = a;
    a_to_p3.FrobeniusMapInPlace(3 * index_multiplier);
    a_to_r_minus_1 *= (a_to_p2 * a_to_p3);

    // Since aʳ, which is |Norm()|, is in the base field, computing the constant
    // is enough.
    BaseField a_to_r = BaseField::Zero();
    a_to_r += a.c1_ * a_to_r_minus_1.c3_;
    a_to_r += a.c2_ * a_to_r_minus_1.c2_;
    a_to_r += a.c3_ * a_to_r_minus_1.c1_;
    a_to_r = Config::MulByNonResidue(a_to_r);
    a_to_r += a.c0_ * a_to_r_minus_1.c0_;

    // a⁻¹ = aʳ⁻¹ * a⁻ʳ
    b = a_to_r_minus_1 * *a_to_r.Inverse();
    return true;
  }

  // c = c0_ + c1_ * X + c2_ * X² + c3_ * X³
  BaseField c0_;
  BaseField c1_;
  BaseField c2_;
  BaseField c3_;

  static BaseField kInv2;
  static BaseField kInv3;
  static BaseField kInv4;
  static BaseField kInv6;
  static BaseField kInv12;
  static BaseField kInv20;
  static BaseField kInv24;
  static BaseField kInv30;
  static BaseField kInv120;
  static BaseField kNeg5;
  static BaseField kNegInv2;
  static BaseField kNegInv3;
  static BaseField kNegInv4;
  static BaseField kNegInv6;
  static BaseField kNegInv12;
  static BaseField kNegInv24;
  static BaseField kNegInv120;
};

#define ADD_STATIC_MEMBER(name)                      \
  template <typename Derived>                        \
  typename QuarticExtensionField<Derived>::BaseField \
      QuarticExtensionField<Derived>::name

ADD_STATIC_MEMBER(kInv2);
ADD_STATIC_MEMBER(kInv3);
ADD_STATIC_MEMBER(kInv4);
ADD_STATIC_MEMBER(kInv6);
ADD_STATIC_MEMBER(kInv12);
ADD_STATIC_MEMBER(kInv20);
ADD_STATIC_MEMBER(kInv24);
ADD_STATIC_MEMBER(kInv30);
ADD_STATIC_MEMBER(kInv120);
ADD_STATIC_MEMBER(kNeg5);
ADD_STATIC_MEMBER(kNegInv2);
ADD_STATIC_MEMBER(kNegInv3);
ADD_STATIC_MEMBER(kNegInv4);
ADD_STATIC_MEMBER(kNegInv6);
ADD_STATIC_MEMBER(kNegInv12);
ADD_STATIC_MEMBER(kNegInv24);
ADD_STATIC_MEMBER(kNegInv120);

#undef ADD_STATIC_MEMBER

template <
    typename BaseField, typename Derived,
    std::enable_if_t<std::is_same_v<BaseField, typename Derived::BaseField>>* =
        nullptr>
Derived operator*(const BaseField& element,
                  const QuarticExtensionField<Derived>& f) {
  return static_cast<const Derived&>(f) * element;
}

}  // namespace math

namespace base {

template <typename Derived>
class Copyable<Derived, std::enable_if_t<std::is_base_of_v<
                            math::QuarticExtensionField<Derived>, Derived>>> {
 public:
  static bool WriteTo(
      const math::QuarticExtensionField<Derived>& quadratic_extension_field,
      Buffer* buffer) {
    return buffer->WriteMany(
        quadratic_extension_field.c0(), quadratic_extension_field.c1(),
        quadratic_extension_field.c2(), quadratic_extension_field.c3());
  }

  static bool ReadFrom(
      const ReadOnlyBuffer& buffer,
      math::QuarticExtensionField<Derived>* quadratic_extension_field) {
    typename Derived::BaseField c0;
    typename Derived::BaseField c1;
    typename Derived::BaseField c2;
    typename Derived::BaseField c3;
    if (!buffer.ReadMany(&c0, &c1, &c2, &c3)) return false;

    *quadratic_extension_field = math::QuarticExtensionField<Derived>(
        std::move(c0), std::move(c1), std::move(c2), std::move(c3));
    return true;
  }

  static size_t EstimateSize(
      const math::QuarticExtensionField<Derived>& quadratic_extension_field) {
    return base::EstimateSize(
        quadratic_extension_field.c0(), quadratic_extension_field.c1(),
        quadratic_extension_field.c2(), quadratic_extension_field.c3());
  }
};

template <typename Derived>
class RapidJsonValueConverter<
    Derived, std::enable_if_t<std::is_base_of_v<
                 math::QuarticExtensionField<Derived>, Derived>>> {
 public:
  using BaseField = typename math::QuarticExtensionField<Derived>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(
      const math::QuarticExtensionField<Derived>& value, Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "c0", value.c0(), allocator);
    AddJsonElement(object, "c1", value.c1(), allocator);
    AddJsonElement(object, "c2", value.c2(), allocator);
    AddJsonElement(object, "c3", value.c3(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::QuarticExtensionField<Derived>* value,
                 std::string* error) {
    BaseField c0;
    BaseField c1;
    BaseField c2;
    BaseField c3;
    if (!ParseJsonElement(json_value, "c0", &c0, error)) return false;
    if (!ParseJsonElement(json_value, "c1", &c1, error)) return false;
    if (!ParseJsonElement(json_value, "c2", &c2, error)) return false;
    if (!ParseJsonElement(json_value, "c3", &c3, error)) return false;
    *value = math::QuarticExtensionField<Derived>(std::move(c0), std::move(c1),
                                                  std::move(c2), std::move(c3));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_QUARTIC_EXTENSION_FIELD_H_
