// clang-format off

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <optional>

#include "absl/base/call_once.h"

#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/prime_field_base.h"
#include "tachyon/math/base/byinverter.h"

extern "C" void %{prefix}_rawAdd(uint64_t result[%{n}], const uint64_t a[%{n}], const uint64_t b[%{n}]);
extern "C" void %{prefix}_rawSub(uint64_t result[%{n}], const uint64_t a[%{n}], const uint64_t b[%{n}]);
extern "C" void %{prefix}_rawNeg(uint64_t result[%{n}], const uint64_t a[%{n}]);
extern "C" void %{prefix}_rawMMul(uint64_t result[%{n}], const uint64_t a[%{n}], const uint64_t b[%{n}]);
extern "C" void %{prefix}_rawMSquare(uint64_t result[%{n}], const uint64_t a[%{n}]);
extern "C" void %{prefix}_rawToMontgomery(uint64_t result[%{n}], const uint64_t a[%{n}]);
extern "C" void %{prefix}_rawFromMontgomery(uint64_t result[%{n}], const uint64_t a[%{n}]);
extern "C" int %{prefix}_rawIsEq(const uint64_t a[%{n}], const uint64_t b[%{n}]);
extern "C" int %{prefix}_rawIsZero(const uint64_t v[%{n}]);

namespace tachyon::math {

template <typename Config>
class PrimeFieldGpu;

template <typename _Config>
class PrimeField<_Config, std::enable_if_t<_Config::%{asm_flag}>> final
    : public PrimeFieldBase<PrimeField<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using value_type = BigInt<N>;

  using CpuField = PrimeField<Config>;
  using GpuField = PrimeFieldGpu<Config>;
  
  constexpr static BYInverter<N> inverter =
      BYInverter<N>(Config::kModulus, Config::kMontgomeryR2);

  constexpr PrimeField() = default;
  template <typename T,
            std::enable_if_t<std::is_constructible_v<BigInt<N>, T>>* = nullptr>
  constexpr explicit PrimeField(T value) : PrimeField(BigInt<N>(value)) {}
  constexpr explicit PrimeField(const BigInt<N>& value) {
    DCHECK_LT(value, Config::kModulus);
    %{prefix}_rawToMontgomery(value_.limbs, value.limbs);
  }
  constexpr PrimeField(const PrimeField& other) = default;
  constexpr PrimeField& operator=(const PrimeField& other) = default;
  constexpr PrimeField(PrimeField&& other) = default;
  constexpr PrimeField& operator=(PrimeField&& other) = default;

  constexpr static PrimeField Zero() { return PrimeField(); }

  constexpr static PrimeField One() {
    PrimeField ret{};
    ret.value_ = Config::kOne;
    return ret;
  }

  constexpr static PrimeField MinusOne() {
    PrimeField ret{};
    ret.value_ = Config::kMinusOne;
    return ret;
  }

  constexpr static PrimeField TwoInv() {
    PrimeField ret{};
    ret.value_ = Config::kTwoInv;
    return ret;
  }

  static PrimeField Random() {
    return PrimeField(BigInt<N>::Random(Config::kModulus));
  }

  constexpr static std::optional<PrimeField> FromDecString(std::string_view str) {
    std::optional<BigInt<N>> value = BigInt<N>::FromDecString(str);
    if (!value.has_value()) return std::nullopt;
    if (value >= Config::kModulus) {
      LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
      return std::nullopt;
    }
    return PrimeField(std::move(value).value());
  }
  constexpr static std::optional<PrimeField> FromHexString(std::string_view str) {
    std::optional<BigInt<N>> value = BigInt<N>::FromHexString(str);
    if (!value.has_value()) return std::nullopt;
    if (value >= Config::kModulus) {
      LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
      return std::nullopt;
    }
    return PrimeField(std::move(value).value());
  }

  constexpr static PrimeField FromBigInt(const BigInt<N>& big_int) {
    return PrimeField(big_int);
  }

  constexpr static PrimeField FromMontgomery(const BigInt<N>& big_int) {
    PrimeField ret{};
    ret.value_ = big_int;
    return ret;
  }

  static PrimeField FromMpzClass(const mpz_class& value) {
    BigInt<N> big_int;
    gmp::CopyLimbs(value, big_int.limbs);
    return FromBigInt(big_int);
  }

  static void Init() {
    static absl::once_flag once;
    absl::call_once(once, []() {
      Config::Init();
      VLOG(1) << Config::kName << " initialized";
    });
  }

  const value_type& value() const { return value_; }

  constexpr bool IsZero() const { return %{prefix}_rawIsZero(value_.limbs); }

  constexpr bool IsOne() const {
    return %{prefix}_rawIsEq(value_.limbs, Config::kOne.limbs);
  }

  constexpr bool IsMinusOne() const {
    return %{prefix}_rawIsEq(value_.limbs, Config::kMinusOne.limbs);
  }

  std::string ToString() const { return ToBigInt().ToString(); }
  std::string ToHexString(bool pad_zero = false) const {
    return ToBigInt().ToHexString(pad_zero);
  }

  mpz_class ToMpzClass() const {
    mpz_class ret;
    gmp::WriteLimbs(ToBigInt().limbs, N, &ret);
    return ret;
  }

  // TODO(chokobole): Support bigendian.
  constexpr BigInt<N> ToBigInt() const {
    BigInt<N> ret;
    %{prefix}_rawFromMontgomery(ret.limbs, value_.limbs);
    return ret;
  }

  constexpr uint64_t& operator[](size_t i) { return value_[i]; }
  constexpr const uint64_t& operator[](size_t i) const { return value_[i]; }

  constexpr bool operator==(const PrimeField& other) const {
    return %{prefix}_rawIsEq(value_.limbs, other.value_.limbs);
  }

  constexpr bool operator!=(const PrimeField& other) const {
    return !operator==(other);
  }

  constexpr bool operator<(const PrimeField& other) const {
    return ToBigInt() < other.ToBigInt();
  }

  constexpr bool operator>(const PrimeField& other) const {
    return ToBigInt() > other.ToBigInt();
  }

  constexpr bool operator<=(const PrimeField& other) const {
    return ToBigInt() <= other.ToBigInt();
  }

  constexpr bool operator>=(const PrimeField& other) const {
    return ToBigInt() >= other.ToBigInt();
  }

  // AdditiveSemigroup methods
  constexpr PrimeField Add(const PrimeField& other) const {
    PrimeField ret{};
    %{prefix}_rawAdd(ret.value_.limbs, value_.limbs, other.value_.limbs);
    return ret;
  }

  constexpr PrimeField& AddInPlace(const PrimeField& other) {
    %{prefix}_rawAdd(value_.limbs, value_.limbs, other.value_.limbs);
    return *this;
  }

  // AdditiveGroup methods
  constexpr PrimeField Sub(const PrimeField& other) const {
    PrimeField ret{};
    %{prefix}_rawSub(ret.value_.limbs, value_.limbs, other.value_.limbs);
    return ret;
  }

  constexpr PrimeField& SubInPlace(const PrimeField& other) {
    %{prefix}_rawSub(value_.limbs, value_.limbs, other.value_.limbs);
    return *this;
  }

  constexpr PrimeField Negate() const {
    PrimeField ret{};
    %{prefix}_rawNeg(ret.value_.limbs, value_.limbs);
    return ret;
  }

  constexpr PrimeField& NegateInPlace() {
    %{prefix}_rawNeg(value_.limbs, value_.limbs);
    return *this;
  }

  // TODO(chokobole): Support bigendian.
  // MultiplicativeSemigroup methods
  constexpr PrimeField Mul(const PrimeField& other) const {
    PrimeField ret{};
    %{prefix}_rawMMul(ret.value_.limbs, value_.limbs, other.value_.limbs);
    return ret;
  }

  constexpr PrimeField& MulInPlace(const PrimeField& other) {
    %{prefix}_rawMMul(value_.limbs, value_.limbs, other.value_.limbs);
    return *this;
  }

  constexpr PrimeField Square() const {
    PrimeField ret{};
    %{prefix}_rawMSquare(ret.value_.limbs, value_.limbs);
    return ret;
  }

  constexpr PrimeField& SquareInPlace() {
    %{prefix}_rawMSquare(value_.limbs, value_.limbs);
    return *this;
  }

  // MultiplicativeGroup methods
  constexpr std::optional<PrimeField> Inverse() const {
    PrimeField ret{};
    if (inverter.Invert(value_, ret.value_)) {
      return ret;
    }
    LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
    return std::nullopt;
  }

  [[nodiscard]] constexpr std::optional<PrimeField*> InverseInPlace() {
    if (inverter.Invert(value_, value_)) {
      return this;
    }
    LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
    return std::nullopt;
  }

 private:
  BigInt<N> value_;
};

}  // namespace tachyon::math

// clang-format on
