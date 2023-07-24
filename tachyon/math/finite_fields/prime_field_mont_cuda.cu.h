#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_CUDA_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_CUDA_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/base/random.h"
#include "tachyon/math/base/arithmetics.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/finite_fields/kernels/carry_chain.cu.h"
#include "tachyon/math/finite_fields/modulus.h"
#include "tachyon/math/finite_fields/prime_field_base.h"
#include "tachyon/math/finite_fields/prime_field_mont.h"

namespace tachyon {
namespace math {

template <typename _Config>
class PrimeFieldMontCuda : public PrimeFieldBase<PrimeFieldMontCuda<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;
  constexpr static size_t N32 = kLimbNums * 2;

  using Config = _Config;
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

  constexpr PrimeFieldMontCuda() = default;
  template <typename T,
            std::enable_if_t<std::is_constructible_v<BigInt<N>, T>>* = nullptr>
  constexpr explicit PrimeFieldMontCuda(T value)
      : PrimeFieldMontCuda(BigInt<N>(value)) {}
  constexpr explicit PrimeFieldMontCuda(const BigInt<N>& value) {
    DCHECK_LT(value, GetModulus());
    PrimeFieldMont<Config> p(value);
    value_ = p.value();
  }
  constexpr PrimeFieldMontCuda(const PrimeFieldMontCuda& other) = default;
  constexpr PrimeFieldMontCuda& operator=(const PrimeFieldMontCuda& other) =
      default;
  constexpr PrimeFieldMontCuda(PrimeFieldMontCuda&& other) = default;
  constexpr PrimeFieldMontCuda& operator=(PrimeFieldMontCuda&& other) = default;

  __host__ __device__ constexpr static PrimeFieldMontCuda Zero() {
    return PrimeFieldMontCuda(BigInt<N>::Zero());
  }

  __host__ __device__ constexpr static PrimeFieldMontCuda One() {
    return PrimeFieldMontCuda(BigInt<N>::One());
  }

  static PrimeFieldMontCuda Random() {
    BigInt<N> big_int;
    for (size_t i = 0; i < N; ++i) {
      big_int.limbs[i] = base::Uniform<uint64_t, uint64_t>(
          0, std::numeric_limits<uint64_t>::max());
    }
    while (big_int >= GetModulus()) {
      big_int.DivBy2InPlace();
    }
    return PrimeFieldMontCuda(big_int);
  }

  constexpr static PrimeFieldMontCuda FromDecString(std::string_view str) {
    return PrimeFieldMontCuda(BigInt<N>::FromDecString(str));
  }
  constexpr static PrimeFieldMontCuda FromHexString(std::string_view str) {
    return PrimeFieldMontCuda(BigInt<N>::FromHexString(str));
  }

  constexpr static PrimeFieldMontCuda FromBigInt(const BigInt<N>& big_int) {
    return PrimeFieldMontCuda(big_int);
  }

  constexpr static PrimeFieldMontCuda FromMontgomery(const BigInt<N>& big_int) {
    PrimeFieldMontCuda ret;
    ret.value_ = big_int;
    return ret;
  }

  __host__ __device__ constexpr static BigInt<N> GetModulus() {
    return Config::kModulus;
  }

  __host__ __device__ const value_type& value() const { return value_; }

  __host__ __device__ constexpr bool IsZero() const {
    const uint64_t* x = value_.limbs;
    uint64_t limbs_or = x[0];
    for (size_t i = 1; i < N; ++i) limbs_or |= x[i];
    return limbs_or == 0;
  }

  __host__ __device__ constexpr bool IsOne() const {
    BigInt<N> value = ToBigInt();
    const uint64_t* x = value.limbs;
    uint64_t limbs_or = 0;
    for (size_t i = 1; i < N; ++i) limbs_or |= x[i];
    return x[0] == 1 && limbs_or == 0;
  }

  __host__ __device__ constexpr bool IsEven() const { return value_.IsEven(); }

  __host__ __device__ constexpr bool IsOdd() const { return value_.IsOdd(); }

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

  __host__ __device__ constexpr uint64_t& operator[](size_t i) {
    return value_[i];
  }
  __host__ __device__ constexpr const uint64_t& operator[](size_t i) const {
    return value_[i];
  }

  __host__ __device__ constexpr bool operator==(
      const PrimeFieldMontCuda& other) const {
    const uint64_t* x = value_.limbs;
    const uint64_t* y = other.value_.limbs;
    uint64_t limbs_or = x[0] ^ y[0];
    for (size_t i = 1; i < N; ++i) limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
  }

  __host__ __device__ constexpr bool operator!=(
      const PrimeFieldMontCuda& other) const {
    return !operator==(other);
  }

  __device__ constexpr bool operator<(const PrimeFieldMontCuda& other) const {
    PrimeFieldMontCuda results;
    uint64_t carry = SubLimbs<true>(value_, other.value_, results.value_);
    return carry;
  }

  __device__ constexpr bool operator>(const PrimeFieldMontCuda& other) const {
    PrimeFieldMontCuda results;
    uint64_t carry = SubLimbs<true>(other.value_, value_, results.value_);
    return carry;
  }

  __device__ constexpr bool operator<=(const PrimeFieldMontCuda& other) const {
    return !operator>(other);
  }

  __device__ constexpr bool operator>=(const PrimeFieldMontCuda& other) const {
    return !operator<(other);
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    return gmp::DivBy2Exp(ToMpzClass(), exp);
  }

  // AdditiveSemigroup methods
  __device__ constexpr PrimeFieldMontCuda& AddInPlace(
      const PrimeFieldMontCuda& other) {
    AddLimbs<false>(value_, other.value_, value_);
    *this = Clamp(*this);
    return *this;
  }

  __device__ constexpr PrimeFieldMontCuda& DoubleInPlace() {
    return AddInPlace(*this);
  }

  // AdditiveGroup methods
  __device__ constexpr PrimeFieldMontCuda& SubInPlace(
      const PrimeFieldMontCuda& other) {
    uint64_t carry = SubLimbs<true>(value_, other.value_, value_);
    if (carry == 0) return *this;
    AddLimbs<false>(value_, GetModulus(), value_);
    return *this;
  }

  __device__ constexpr PrimeFieldMontCuda& NegInPlace() {
    BigInt<N> result;
    SubLimbs<false>(GetModulus(), value_, result);
    value_ = result;
    return *this;
  }

  // MultiplicativeSemigroup methods
  __device__ constexpr PrimeFieldMontCuda& MulInPlace(
      const PrimeFieldMontCuda& other) {
    // Forces us to think more carefully about the last carry bit if we use a
    // modulus with fewer than 2 leading zeroes of slack.
    static_assert(!(Config::kModulus[N - 1] >> 62));
    BigInt<N> result;
    MulLimbs(value_, other.value_, result);
    value_ = result;
    *this = Clamp(*this);
    return *this;
  }

  __device__ constexpr PrimeFieldMontCuda& SquareInPlace() {
    return MulInPlace(*this);
  }

  // MultiplicativeGroup methods
  __device__ constexpr PrimeFieldMontCuda& InverseInPlace() {
    if (IsZero()) return *this;

    BigInt<N> u = value_;
    BigInt<N> v = GetModulus();
    PrimeFieldMontCuda b;
    b.value_ = kMontgomeryR2;
    // NOTE(chokobole): Do not use this.
    // PrimeFieldMontCuda c = PrimeFieldMontCuda::Zero();
    PrimeFieldMontCuda c;

    while (!u.IsOne() && !v.IsOne()) {
      while (u.IsEven()) {
        u = DivBy2Limbs(u);
        if (b.IsOdd()) AddLimbs<false>(b.value_, GetModulus(), b.value_);
        b.value_ = DivBy2Limbs(b.value_);
      }

      while (v.IsEven()) {
        v = DivBy2Limbs(v);
        if (c.IsOdd()) AddLimbs<false>(c.value_, GetModulus(), c.value_);
        c.value_ = DivBy2Limbs(c.value_);
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
  __device__ constexpr static uint64_t AddLimbs(const BigInt<N>& xs,
                                                const BigInt<N>& ys,
                                                BigInt<N>& results) {
    const uint64_t* x = xs.limbs;
    const uint64_t* y = ys.limbs;
    uint64_t* r = results.limbs;
    kernels::u64::CarryChain<CarryOut ? N + 1 : N> chain;
    for (size_t i = 0; i < N; ++i) {
      r[i] = chain.Add(x[i], y[i]);
    }
    if constexpr (!CarryOut) return 0;
    return chain.Add(0, 0);
  }

  template <bool CarryOut>
  __device__ constexpr static uint64_t SubLimbs(const BigInt<N>& xs,
                                                const BigInt<N>& ys,
                                                BigInt<N>& results) {
    const uint64_t* x = xs.limbs;
    const uint64_t* y = ys.limbs;
    uint64_t* r = results.limbs;
    kernels::u64::CarryChain<CarryOut ? N + 1 : N> chain;
    for (size_t i = 0; i < N; ++i) {
      r[i] = chain.Sub(x[i], y[i]);
    }
    if constexpr (!CarryOut) return 0;
    return chain.Sub(0, 0);
  }

  __device__ constexpr static void MulLimbs(const BigInt<N>& xs,
                                            const BigInt<N>& ys,
                                            BigInt<N>& results) {
    constexpr uint32_t n = N32;
    const uint32_t* x = reinterpret_cast<const uint32_t*>(xs.limbs);
    const uint32_t* y = reinterpret_cast<const uint32_t*>(ys.limbs);
    uint32_t* even = reinterpret_cast<uint32_t*>(results.limbs);

    ALIGNAS(8)
    uint32_t odd[n + 1] = {
        0,
    };
    size_t i;
    for (i = 0; i < n; i += 2) {
      MadNRedc(&even[0], &odd[0], x, y[i], i == 0);
      MadNRedc(&odd[0], &even[0], x, y[i + 1]);
    }

    // merge |even| and |odd|
    even[0] = ptx::u32::AddCc(even[0], odd[1]);
    for (i = 1; i < n - 1; ++i) {
      even[i] = ptx::u32::AddcCc(even[i], odd[i + 1]);
    }
    even[i] = ptx::u32::Addc(even[i], 0);

    // final reduction from [0, 2 * mod) to [0, mod) not done here, instead
    // performed optionally in MulInPlace.
  }

  __device__ constexpr static BigInt<N> DivBy2Limbs(const BigInt<N>& xs) {
    BigInt<N> results;
    const uint32_t* x = reinterpret_cast<const uint32_t*>(xs.limbs);
    uint32_t* r = reinterpret_cast<uint32_t*>(results.limbs);
    for (size_t i = 0; i < N32 - 1; ++i) {
      r[i] = static_cast<uint32_t>(__funnelshift_rc(x[i], x[i + 1], 1));
    }
    r[N32 - 1] = x[N32 - 1] >> 1;
    return results;
  }

  // The following algorithms are adaptations of
  // http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf,
  // taken from https://github.com/z-prize/test-msm-gpu (under Apache 2.0
  // license) and modified to use our datatypes. We had our own implementation
  // of http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf, but the
  // sppark versions achieved lower instruction count thanks to clever carry
  // handling, so we decided to just use theirs.

  __device__ constexpr static void MulN(uint32_t* acc, const uint32_t* a,
                                        uint32_t bi, size_t n = N32) {
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = ptx::u32::MulLo(a[i], bi);
      acc[i + 1] = ptx::u32::MulHi(a[i], bi);
    }
  }

  __device__ constexpr static void CMadN(uint32_t* acc, const uint32_t* a,
                                         uint32_t bi, size_t n = N32) {
    acc[0] = ptx::u32::MadLoCc(a[0], bi, acc[0]);
    acc[1] = ptx::u32::MadcHiCc(a[0], bi, acc[1]);
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::u32::MadcLoCc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::u32::MadcHiCc(a[i], bi, acc[i + 1]);
    }
  }

  __device__ constexpr static void MadcNRshift(uint32_t* odd, const uint32_t* a,
                                               uint32_t bi) {
    constexpr uint32_t n = N32;
    for (size_t i = 0; i < n - 2; i += 2) {
      odd[i] = ptx::u32::MadcLoCc(a[i], bi, odd[i + 2]);
      odd[i + 1] = ptx::u32::MadcHiCc(a[i], bi, odd[i + 3]);
    }
    odd[n - 2] = ptx::u32::MadcLoCc(a[n - 2], bi, 0);
    odd[n - 1] = ptx::u32::MadcHi(a[n - 2], bi, 0);
  }

  __device__ constexpr static void MadNRedc(uint32_t* even, uint32_t* odd,
                                            const uint32_t* a, uint32_t bi,
                                            bool first = false) {
    constexpr uint32_t n = N32;
    const uint32_t* const modulus =
        reinterpret_cast<const uint32_t* const>(GetModulus().limbs);
    if (first) {
      MulN(odd, a + 1, bi);
      MulN(even, a, bi);
    } else {
      even[0] = ptx::u32::AddCc(even[0], odd[1]);
      MadcNRshift(odd, a + 1, bi);
      CMadN(even, a, bi);
      odd[n - 1] = ptx::u32::Addc(odd[n - 1], 0);
    }
    uint32_t mi = even[0] * kInverse32;
    CMadN(odd, modulus + 1, mi);
    CMadN(even, modulus, mi);
    odd[n - 1] = ptx::u32::Addc(odd[n - 1], 0);
  }

  __device__ constexpr static PrimeFieldMontCuda Clamp(PrimeFieldMontCuda& xs) {
    PrimeFieldMontCuda results;
    return SubLimbs<true>(xs.value_, GetModulus(), results.value_) ? xs
                                                                   : results;
  }

  BigInt<N> value_;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os,
                         const PrimeFieldMontCuda<Config>& f) {
  return os << f.ToString();
}

template <typename Config>
class MultiplicativeIdentity<PrimeFieldMontCuda<Config>> {
 public:
  using F = PrimeFieldMontCuda<Config>;

  static const F& One() {
    static F one(F::One());
    return one;
  }

  constexpr static bool IsOne(const F& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<PrimeFieldMontCuda<Config>> {
 public:
  using F = PrimeFieldMontCuda<Config>;

  static const F& Zero() {
    static F zero(F::Zero());
    return zero;
  }

  constexpr static bool IsZero(const F& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_CUDA_H_
