#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CUDA_DEBUG_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CUDA_DEBUG_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "tachyon/base/random.h"
#include "tachyon/math/base/arithmetics.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/finite_fields/carry_chain.h"
#include "tachyon/math/finite_fields/modulus.h"
#include "tachyon/math/finite_fields/prime_field.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::math {

template <typename _Config>
class PrimeFieldCudaDebug
    : public PrimeFieldBase<PrimeFieldCudaDebug<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;
  constexpr static size_t N32 = kLimbNums * 2;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using value_type = BigInt<N>;

  constexpr static bool kModulusHasSpareBit =
      Modulus<N>::HasSpareBit(Config::kModulus);
  constexpr static bool kCanUseNoCarryMulOptimization =
      Modulus<N>::CanUseNoCarryMulOptimization(Config::kModulus);
  constexpr static BigInt<N> kMontgomeryR =
      Modulus<N>::MontgomeryR(Config::kModulus);
  constexpr static BigInt<N> kMontgomeryR2 =
      Modulus<N>::MontgomeryR2(Config::kModulus);
  constexpr static uint64_t kInverse64 =
      Modulus<N>::template Inverse<uint64_t>(Config::kModulus);
  constexpr static uint32_t kInverse32 =
      Modulus<N>::template Inverse<uint32_t>(Config::kModulus);

  constexpr PrimeFieldCudaDebug() = default;
  template <typename T,
            std::enable_if_t<std::is_constructible_v<BigInt<N>, T>>* = nullptr>
  constexpr explicit PrimeFieldCudaDebug(T value)
      : PrimeFieldCudaDebug(BigInt<N>(value)) {}
  constexpr explicit PrimeFieldCudaDebug(const BigInt<N>& value) {
    DCHECK_LT(value, Config::kModulus);
    PrimeField<Config> p(value);
    value_ = p.value();
  }
  constexpr PrimeFieldCudaDebug(const PrimeFieldCudaDebug& other) = default;
  constexpr PrimeFieldCudaDebug& operator=(const PrimeFieldCudaDebug& other) =
      default;
  constexpr PrimeFieldCudaDebug(PrimeFieldCudaDebug&& other) = default;
  constexpr PrimeFieldCudaDebug& operator=(PrimeFieldCudaDebug&& other) =
      default;

  constexpr static PrimeFieldCudaDebug Zero() { return PrimeFieldCudaDebug(); }

  constexpr static PrimeFieldCudaDebug One() {
    PrimeFieldCudaDebug ret;
    ret.value_ = Config::kOne;
    return ret;
  }

  static PrimeFieldCudaDebug Random() {
    PrimeFieldCudaDebug ret;
    ret.value_ = PrimeField<Config>::Random().value();
    return ret;
  }

  constexpr static PrimeFieldCudaDebug FromDecString(std::string_view str) {
    return PrimeFieldCudaDebug(BigInt<N>::FromDecString(str));
  }
  constexpr static PrimeFieldCudaDebug FromHexString(std::string_view str) {
    return PrimeFieldCudaDebug(BigInt<N>::FromHexString(str));
  }

  constexpr static PrimeFieldCudaDebug FromBigInt(const BigInt<N>& big_int) {
    return PrimeFieldCudaDebug(big_int);
  }

  constexpr static PrimeFieldCudaDebug FromMontgomery(
      const BigInt<N>& big_int) {
    PrimeFieldCudaDebug ret;
    ret.value_ = big_int;
    return ret;
  }

  static PrimeFieldCudaDebug FromMpzClass(const mpz_class& value) {
    BigInt<N> big_int;
    gmp::CopyLimbs(value, big_int.limbs);
    return FromBigInt(big_int);
  }

  static void Init() {
    // Do nothing.
  }

  const value_type& value() const { return value_; }

  constexpr bool IsZero() const { return ToBigInt().IsZero(); }

  constexpr bool IsOne() const { return ToBigInt().IsOne(); }

  constexpr bool IsEven() const { return value_.IsEven(); }

  constexpr bool IsOdd() const { return value_.IsOdd(); }

  std::string ToString() const { return ToBigInt().ToString(); }
  std::string ToHexString() const { return ToBigInt().ToHexString(); }

  mpz_class ToMpzClass() const {
    mpz_class ret;
    gmp::WriteLimbs(ToBigInt().limbs, N, &ret);
    return ret;
  }

  constexpr BigInt<N> ToBigInt() const {
    return BigInt<N>::FromMontgomery64(value_, Config::kModulus, kInverse64);
  }

  constexpr const BigInt<N>& ToMontgomery() const { return value_; }

  constexpr uint64_t& operator[](size_t i) { return value_[i]; }
  constexpr const uint64_t& operator[](size_t i) const { return value_[i]; }

  constexpr bool operator==(const PrimeFieldCudaDebug& other) const {
    return ToBigInt() == other.ToBigInt();
  }

  constexpr bool operator!=(const PrimeFieldCudaDebug& other) const {
    return ToBigInt() != other.ToBigInt();
  }

  constexpr bool operator<(const PrimeFieldCudaDebug& other) const {
    return ToBigInt() < other.ToBigInt();
  }

  constexpr bool operator>(const PrimeFieldCudaDebug& other) const {
    return ToBigInt() > other.ToBigInt();
  }

  constexpr bool operator<=(const PrimeFieldCudaDebug& other) const {
    return ToBigInt() <= other.ToBigInt();
  }

  constexpr bool operator>=(const PrimeFieldCudaDebug& other) const {
    return ToBigInt() >= other.ToBigInt();
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    return gmp::DivBy2Exp(ToMpzClass(), exp);
  }

  // AdditiveSemigroup methods
  constexpr PrimeFieldCudaDebug& AddInPlace(const PrimeFieldCudaDebug& other) {
    AddLimbs<false>(value_, other.value_, value_);
    *this = Clamp(*this);
    return *this;
  }

  constexpr PrimeFieldCudaDebug& DoubleInPlace() { return AddInPlace(*this); }

  // AdditiveGroup methods
  constexpr PrimeFieldCudaDebug& SubInPlace(const PrimeFieldCudaDebug& other) {
    uint64_t carry = SubLimbs<true>(value_, other.value_, value_);
    if (carry == 0) return *this;
    AddLimbs<false>(value_, Config::kModulus, value_);
    return *this;
  }

  constexpr PrimeFieldCudaDebug& NegInPlace() {
    BigInt<N> result;
    SubLimbs<false>(Config::kModulus, value_, result);
    value_ = result;
    return *this;
  }

  // MultiplicativeSemigroup methods
  constexpr PrimeFieldCudaDebug& MulInPlace(const PrimeFieldCudaDebug& other) {
    // Forces us to think more carefully about the last carry bit if we use a
    // modulus with fewer than 2 leading zeroes of slack.
    static_assert(!(Config::kModulus[N - 1] >> 62));
    BigInt<N> result;
    MulLimbs(value_, other.value_, result);
    value_ = result;
    *this = Clamp(*this);
    return *this;
  }

  constexpr PrimeFieldCudaDebug& SquareInPlace() { return MulInPlace(*this); }

  // MultiplicativeGroup methods
  // TODO(chokobole): Share codes with PrimeField and PrimeFieldCuda.
  constexpr PrimeFieldCudaDebug& InverseInPlace() {
    CHECK(!IsZero());

    BigInt<N> u = value_;
    BigInt<N> v = Config::kModulus;
    PrimeFieldCudaDebug b;
    b.value_ = kMontgomeryR2;
    PrimeFieldCudaDebug c = PrimeFieldCudaDebug::Zero();

    while (!u.IsOne() && !v.IsOne()) {
      while (u.IsEven()) {
        u.DivBy2InPlace();
        if (b.IsOdd()) AddLimbs<false>(b.value_, Config::kModulus, b.value_);
        b.value_.DivBy2InPlace();
      }

      while (v.IsEven()) {
        v.DivBy2InPlace();
        if (c.IsOdd()) AddLimbs<false>(c.value_, Config::kModulus, c.value_);
        c.value_.DivBy2InPlace();
      }
      if (v < u) {
        SubLimbs<false>(u, v, u);
        b -= c;
      } else {
        SubLimbs<false>(v, u, v);
        c -= b;
      }
    }
    if (u.IsOne()) {
      *this = b;
    } else {
      *this = c;
    }
    return *this;
  }

 private:
  template <bool CarryOut>
  constexpr static uint64_t AddLimbs(const BigInt<N>& xs, const BigInt<N>& ys,
                                     BigInt<N>& results) {
    const uint64_t* x = xs.limbs;
    const uint64_t* y = ys.limbs;
    uint64_t* r = results.limbs;
    u64::CarryChain<CarryOut ? N + 1 : N> chain;
    AddResult<uint64_t> result;
    for (size_t i = 0; i < N; ++i) {
      result = chain.Add(x[i], y[i], result.carry);
      r[i] = result.result;
    }
    if constexpr (!CarryOut) return 0;
    return chain.Add(0, 0, result.carry).result;
  }

  template <bool CarryOut>
  constexpr static uint64_t SubLimbs(const BigInt<N>& xs, const BigInt<N>& ys,
                                     BigInt<N>& results) {
    const uint64_t* x = xs.limbs;
    const uint64_t* y = ys.limbs;
    uint64_t* r = results.limbs;
    u64::CarryChain<CarryOut ? N + 1 : N> chain;
    SubResult<uint64_t> result;
    for (size_t i = 0; i < N; ++i) {
      result = chain.Sub(x[i], y[i], result.borrow);
      r[i] = result.result;
    }
    if constexpr (!CarryOut) return 0;
    return chain.Sub(0, 0, result.borrow).result;
  }

  constexpr static void MulLimbs(const BigInt<N>& xs, const BigInt<N>& ys,
                                 BigInt<N>& results) {
    constexpr uint32_t n = N32;
    const uint32_t* x = reinterpret_cast<const uint32_t*>(xs.limbs);
    const uint32_t* y = reinterpret_cast<const uint32_t*>(ys.limbs);
    BigInt<N + 1> results2;
    uint32_t* even = reinterpret_cast<uint32_t*>(results.limbs);
    uint32_t* odd = reinterpret_cast<uint32_t*>(results2.limbs);
    size_t i = 0;
    for (i = 0; i < n; i += 2) {
      MadNRedc(&even[0], &odd[0], x, y[i], i == 0);
      MadNRedc(&odd[0], &even[0], x, y[i + 1]);
    }

    BigInt<N> l;
    uint32_t* l_tmp = reinterpret_cast<uint32_t*>(l.limbs);
    for (size_t i = 1; i < 9; ++i) {
      l_tmp[i - 1] = odd[i];
    }
    // merge |even| and |odd|
    AddResult<uint32_t> result;
    result = internal::u32::AddCc(even[0], odd[1]);
    even[0] = result.result;
    for (i = 1; i < n - 1; ++i) {
      result = internal::u32::AddcCc(even[i], odd[i + 1], result.carry);
      even[i] = result.result;
    }
    result = internal::u32::Addc(even[i], 0, result.carry);
    even[i] = result.result;
    // final reduction from [0, 2 * mod) to [0, mod) not done here, instead
    // performed optionally in MulInPlace.
  }

  constexpr static void MulN(uint32_t* acc, const uint32_t* a, uint32_t bi,
                             size_t n = N32) {
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = internal::u32::MulLo(a[i], bi);
      acc[i + 1] = internal::u32::MulHi(a[i], bi);
    }
  }

  constexpr static uint32_t CMadN(uint32_t* acc, const uint32_t* a, uint32_t bi,
                                  size_t n = N32) {
    AddResult<uint32_t> result;
    result = internal::u32::MadLoCc(a[0], bi, acc[0]);
    acc[0] = result.result;
    result = internal::u32::MadcHiCc(a[0], bi, acc[1], result.carry);
    acc[1] = result.result;
    for (size_t i = 2; i < n; i += 2) {
      result = internal::u32::MadcLoCc(a[i], bi, acc[i], result.carry);
      acc[i] = result.result;
      result = internal::u32::MadcHiCc(a[i], bi, acc[i + 1], result.carry);
      acc[i + 1] = result.result;
    }
    return result.carry;
  }

  constexpr static void MadcNRshift(uint32_t* odd, const uint32_t* a,
                                    uint32_t bi, uint32_t carry) {
    constexpr uint32_t n = N32;
    AddResult<uint32_t> result;
    result.carry = carry;
    for (size_t i = 0; i < n - 2; i += 2) {
      result = internal::u32::MadcLoCc(a[i], bi, odd[i + 2], result.carry);
      odd[i] = result.result;
      result = internal::u32::MadcHiCc(a[i], bi, odd[i + 3], result.carry);
      odd[i + 1] = result.result;
    }
    result = internal::u32::MadcLoCc(a[n - 2], bi, 0, result.carry);
    odd[n - 2] = result.result;
    result = internal::u32::MadcHi(a[n - 2], bi, 0, result.carry);
    odd[n - 1] = result.result;
    CHECK_EQ(result.carry, static_cast<uint32_t>(0));
  }

  constexpr static void MadNRedc(uint32_t* even, uint32_t* odd,
                                 const uint32_t* a, uint32_t bi,
                                 bool first = false) {
    constexpr uint32_t n = N32;
    const uint32_t* const modulus =
        reinterpret_cast<const uint32_t* const>(Config::kModulus.limbs);
    if (first) {
      MulN(odd, a + 1, bi);
      MulN(even, a, bi);
    } else {
      AddResult<uint32_t> result;
      result = internal::u32::AddCc(even[0], odd[1]);
      even[0] = result.result;
      MadcNRshift(odd, a + 1, bi, result.carry);
      uint32_t carry = CMadN(even, a, bi);
      result = internal::u32::Addc(odd[n - 1], 0, carry);
      odd[n - 1] = result.result;
      CHECK_EQ(result.carry, static_cast<uint32_t>(0));
    }
    uint32_t mi = even[0] * kInverse32;
    uint32_t carry = CMadN(odd, modulus + 1, mi);
    CHECK_EQ(carry, static_cast<uint32_t>(0));
    carry = CMadN(even, modulus, mi);
    AddResult<uint32_t> result = internal::u32::Addc(odd[n - 1], 0, carry);
    odd[n - 1] = result.result;
    CHECK_EQ(result.carry, static_cast<uint32_t>(0));
  }

  constexpr static PrimeFieldCudaDebug Clamp(PrimeFieldCudaDebug& xs) {
    PrimeFieldCudaDebug results;
    return SubLimbs<true>(xs.value_, Config::kModulus, results.value_)
               ? xs
               : results;
  }

  BigInt<N> value_;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os,
                         const PrimeFieldCudaDebug<Config>& f) {
  return os << f.ToString();
}

template <typename Config>
class MultiplicativeIdentity<PrimeFieldCudaDebug<Config>> {
 public:
  using F = PrimeFieldCudaDebug<Config>;

  static const F& One() {
    static F one(F::One());
    return one;
  }

  constexpr static bool IsOne(const F& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<PrimeFieldCudaDebug<Config>> {
 public:
  using F = PrimeFieldCudaDebug<Config>;

  static const F& Zero() {
    static F zero(F::Zero());
    return zero;
  }

  constexpr static bool IsZero(const F& value) { return value.IsZero(); }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CUDA_DEBUG_H_
