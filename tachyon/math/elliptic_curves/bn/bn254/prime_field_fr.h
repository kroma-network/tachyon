#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_PRIME_FIELD_FR_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_PRIME_FIELD_FR_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

extern "C" void Fr_rawAdd(uint64_t result[4], const uint64_t a[4],
                          const uint64_t b[4]);
extern "C" void Fr_rawSub(uint64_t result[4], const uint64_t a[4],
                          const uint64_t b[4]);
extern "C" void Fr_rawNeg(uint64_t result[4], const uint64_t a[4]);
extern "C" void Fr_rawMMul(uint64_t result[4], const uint64_t a[4],
                           const uint64_t b[4]);
extern "C" void Fr_rawMSquare(uint64_t result[4], const uint64_t a[4]);
extern "C" void Fr_rawMMul1(uint64_t result[4], const uint64_t a[4],
                            uint64_t b);
extern "C" void Fr_rawToMontgomery(uint64_t result[4], const uint64_t a[4]);
extern "C" void Fr_rawFromMontgomery(uint64_t result[4], const uint64_t a[4]);
extern "C" int Fr_rawIsEq(const uint64_t a[4], const uint64_t b[4]);
extern "C" int Fr_rawIsZero(const uint64_t v[4]);

namespace tachyon::math {

template <typename Config>
class PrimeFieldGpu;

template <typename _Config>
class PrimeField<_Config, std::enable_if_t<_Config::kIsBn254Fr>> final
    : public PrimeFieldBase<PrimeField<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using MontgomeryTy = BigInt<N>;
  using value_type = BigInt<N>;

  using CpuField = PrimeField<Config>;
  using GpuField = PrimeFieldGpu<Config>;

  constexpr PrimeField() = default;
  template <typename T,
            std::enable_if_t<std::is_constructible_v<BigInt<N>, T>>* = nullptr>
  constexpr explicit PrimeField(T value) : PrimeField(BigInt<N>(value)) {}
  constexpr explicit PrimeField(const BigInt<N>& value) {
    DCHECK_LT(value, Config::kModulus);
    Fr_rawToMontgomery(value_.limbs, value.limbs);
  }
  constexpr PrimeField(const PrimeField& other) = default;
  constexpr PrimeField& operator=(const PrimeField& other) = default;
  constexpr PrimeField(PrimeField&& other) = default;
  constexpr PrimeField& operator=(PrimeField&& other) = default;

  constexpr static PrimeField Zero() { return PrimeField(); }

  constexpr static PrimeField One() {
    PrimeField ret;
    ret.value_ = Config::kOne;
    return ret;
  }

  static PrimeField Random() {
    return PrimeField(BigInt<N>::Random(Config::kModulus));
  }

  constexpr static PrimeField FromDecString(std::string_view str) {
    return PrimeField(BigInt<N>::FromDecString(str));
  }
  constexpr static PrimeField FromHexString(std::string_view str) {
    return PrimeField(BigInt<N>::FromHexString(str));
  }

  constexpr static PrimeField FromBigInt(const BigInt<N>& big_int) {
    return PrimeField(big_int);
  }

  constexpr static PrimeField FromMontgomery(const BigInt<N>& big_int) {
    PrimeField ret;
    ret.value_ = big_int;
    return ret;
  }

  static PrimeField FromMpzClass(const mpz_class& value) {
    BigInt<N> big_int;
    gmp::CopyLimbs(value, big_int.limbs);
    return FromBigInt(big_int);
  }

  static void Init() { VLOG(1) << Config::kName << " initialized"; }

  const value_type& value() const { return value_; }

  constexpr bool IsZero() const { return Fr_rawIsZero(value_.limbs); }

  constexpr bool IsOne() const {
    return Fr_rawIsEq(value_.limbs, Config::kOne.limbs);
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
    Fr_rawFromMontgomery(ret.limbs, value_.limbs);
    return ret;
  }

  constexpr const BigInt<N>& ToMontgomery() const { return value_; }

  constexpr uint64_t& operator[](size_t i) { return value_[i]; }
  constexpr const uint64_t& operator[](size_t i) const { return value_[i]; }

  constexpr bool operator==(const PrimeField& other) const {
    return Fr_rawIsEq(value_.limbs, other.value_.limbs);
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

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    return gmp::DivBy2Exp(ToMpzClass(), exp);
  }

  // AdditiveSemigroup methods
  constexpr PrimeField& AddInPlace(const PrimeField& other) {
    PrimeField ret;
    Fr_rawAdd(ret.value_.limbs, value_.limbs, other.value_.limbs);
    *this = ret;
    return *this;
  }

  constexpr PrimeField& DoubleInPlace() { return AddInPlace(*this); }

  // AdditiveGroup methods
  constexpr PrimeField& SubInPlace(const PrimeField& other) {
    PrimeField ret;
    Fr_rawSub(ret.value_.limbs, value_.limbs, other.value_.limbs);
    *this = ret;
    return *this;
  }

  constexpr PrimeField& NegInPlace() {
    PrimeField ret;
    Fr_rawNeg(ret.value_.limbs, value_.limbs);
    *this = ret;
    return *this;
  }

  // TODO(chokobole): Support bigendian.
  // MultiplicativeSemigroup methods
  constexpr PrimeField& MulInPlace(const PrimeField& other) {
    PrimeField ret;
    Fr_rawMMul(ret.value_.limbs, value_.limbs, other.value_.limbs);
    *this = ret;
    return *this;
  }

  constexpr PrimeField& SquareInPlace() {
    PrimeField ret;
    Fr_rawMSquare(ret.value_.limbs, value_.limbs);
    *this = ret;
    return *this;
  }

  // MultiplicativeGroup methods
  PrimeField& DivInPlace(const PrimeField& other) {
    return MulInPlace(other.Inverse());
  }

  constexpr PrimeField& InverseInPlace() {
    value_ = value_.template MontgomeryInverse<Config::kModulusHasSpareBit>(
        Config::kModulus, Config::kMontgomeryR2);
    return *this;
  }

 private:
  BigInt<N> value_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_PRIME_FIELD_FR_H_
