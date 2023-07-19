#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "tachyon/base/random.h"
#include "tachyon/math/base/arithmetics.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/finite_fields/modulus.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon {
namespace math {

template <typename _Config>
class PrimeFieldMont : public PrimeFieldBase<PrimeFieldMont<_Config>> {
 public:
  static constexpr size_t kModulusBits = _Config::kModulusBits;
  static constexpr size_t kLimbNums = (kModulusBits + 63) / 64;
  static constexpr size_t N = kLimbNums;

  using Config = _Config;
  using value_type = BigInt<N>;

  static constexpr bool kModulusHasSparseBit =
      Modulus<N>::HasSparseBit(Config::kModulus);
  static constexpr bool kCanUseNoCarryMulOptimization =
      Modulus<N>::CanUseNoCarryMulOptimization(Config::kModulus);
  static constexpr BigInt<N> kMontgomeryR =
      Modulus<N>::MontgomeryR(Config::kModulus);
  static constexpr BigInt<N> kMontgomeryR2 =
      Modulus<N>::MontgomeryR2(Config::kModulus);
  static constexpr uint64_t kInverse = Modulus<N>::Inverse(Config::kModulus);

  constexpr PrimeFieldMont() = default;
  template <typename T,
            std::enable_if_t<std::is_constructible_v<BigInt<N>, T>>* = nullptr>
  constexpr explicit PrimeFieldMont(T value)
      : PrimeFieldMont(BigInt<N>(value)) {}
  constexpr explicit PrimeFieldMont(const BigInt<N>& value) : value_(value) {
    DCHECK_LT(value, Config::kModulus);
    PrimeFieldMont p;
    p.value_ = kMontgomeryR2;
    MulInPlace(p);
  }
  constexpr PrimeFieldMont(const PrimeFieldMont& other) = default;
  constexpr PrimeFieldMont& operator=(const PrimeFieldMont& other) = default;
  constexpr PrimeFieldMont(PrimeFieldMont&& other) = default;
  constexpr PrimeFieldMont& operator=(PrimeFieldMont&& other) = default;

  constexpr static PrimeFieldMont Zero() {
    return PrimeFieldMont(BigInt<N>::Zero());
  }

  constexpr static PrimeFieldMont One() {
    return PrimeFieldMont(BigInt<N>::One());
  }

  static PrimeFieldMont Random() {
    BigInt<N> big_int;
    for (size_t i = 0; i < N; ++i) {
      big_int.limbs[i] = base::Uniform<uint64_t, uint64_t>(
          0, std::numeric_limits<uint64_t>::max());
    }
    while (big_int >= Config::kModulus) {
      big_int.DivBy2InPlace();
    }
    return PrimeFieldMont(big_int);
  }

  constexpr static PrimeFieldMont FromDecString(std::string_view str) {
    return PrimeFieldMont(BigInt<N>::FromDecString(str));
  }
  constexpr static PrimeFieldMont FromHexString(std::string_view str) {
    return PrimeFieldMont(BigInt<N>::FromHexString(str));
  }

  template <typename T>
  constexpr static PrimeFieldMont FromDevice(const T& field_device) {
    return PrimeFieldMont(field_device.ToBigInt());
  }

  const value_type& value() const { return value_; }

  constexpr bool IsZero() const { return value_.IsZero(); }

  constexpr bool IsOne() const { return ToBigInt().IsOne(); }

  std::string ToString() const { return ToBigInt().ToString(); }
  std::string ToHexString() const { return ToBigInt().ToHexString(); }

  mpz_class ToMpzClass() const {
    mpz_class ret;
    gmp::WriteLimbs(ToBigInt().limbs, N, &ret);
    return ret;
  }

  // TODO(chokobole): Support bigendian.
  constexpr BigInt<N> ToBigInt() const {
    BigInt<N> r = value_;
    // Montgomery Reduction
    for (size_t i = 0; i < N; ++i) {
      uint64_t k = r[i] * kInverse;
      MulResult<uint64_t> result =
          internal::MulAddWithCarry(r[i], k, Config::kModulus[0]);
      for (size_t j = 1; j < N; ++j) {
        result = internal::MulAddWithCarry(r[(j + i) % N], k,
                                           Config::kModulus[j], result.hi);
        r[(j + i) % N] = result.lo;
      }
      r[i] = result.hi;
    }
    return r;
  }

  constexpr bool operator==(const PrimeFieldMont& other) const {
    return ToBigInt() == other.ToBigInt();
  }

  constexpr bool operator!=(const PrimeFieldMont& other) const {
    return ToBigInt() != other.ToBigInt();
  }

  constexpr bool operator<(const PrimeFieldMont& other) const {
    return ToBigInt() < other.ToBigInt();
  }

  constexpr bool operator>(const PrimeFieldMont& other) const {
    return ToBigInt() > other.ToBigInt();
  }

  constexpr bool operator<=(const PrimeFieldMont& other) const {
    return ToBigInt() <= other.ToBigInt();
  }

  constexpr bool operator>=(const PrimeFieldMont& other) const {
    return ToBigInt() >= other.ToBigInt();
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    return gmp::DivBy2Exp(ToMpzClass(), exp);
  }

  // AdditiveSemigroup methods
  constexpr PrimeFieldMont& AddInPlace(const PrimeFieldMont& other) {
    uint64_t carry = 0;
    value_.AddInPlace(other.value_, carry);
    return Clamp(carry);
  }

  constexpr PrimeFieldMont& DoubleInPlace() {
    uint64_t carry = 0;
    value_.MulBy2InPlace(carry);
    return Clamp(carry);
  }

  // AdditiveGroup methods
  constexpr PrimeFieldMont& SubInPlace(const PrimeFieldMont& other) {
    if (other.value_ > value_) {
      value_.AddInPlace(Config::kModulus);
    }
    value_.SubInPlace(other.value_);
    return *this;
  }

  constexpr PrimeFieldMont& NegInPlace() {
    if (!IsZero()) {
      BigInt<N> tmp(Config::kModulus);
      tmp.SubInPlace(value_);
      value_ = tmp;
    }
    return *this;
  }

  // TODO(chokobole): Support bigendian.
  // MultiplicativeSemigroup methods
  constexpr PrimeFieldMont& MulInPlace(const PrimeFieldMont& other) {
    if constexpr (kCanUseNoCarryMulOptimization) {
      BigInt<N> r;
      for (size_t i = 0; i < N; ++i) {
        MulResult<uint64_t> result;
        result = internal::MulAddWithCarry(r.limbs[0], value_.limbs[0],
                                           other.value_.limbs[0]);
        r.limbs[0] = result.lo;

        uint64_t k = r.limbs[0] * kInverse;
        MulResult<uint64_t> result2;
        result2 =
            internal::MulAddWithCarry(r.limbs[0], k, Config::kModulus.limbs[0]);

        for (size_t j = 1; j < N; ++j) {
          result = internal::MulAddWithCarry(r[j], value_.limbs[j],
                                             other.value_.limbs[i], result.hi);
          r[j] = result.lo;
          result2 = internal::MulAddWithCarry(
              r[j], k, Config::kModulus.limbs[j], result2.hi);
          r[j - 1] = result2.lo;
        }
        r[N - 1] = result.hi + result2.hi;
      }
      value_ = r;
      return Clamp(0);
    } else {
      // Alternative implementation, CIOS.
      uint64_t carry = 0;
      MulWithoutConditionSubtract(other, carry);
      // NOTE(chokobole): This is different from arkworks!
      // Seems like arkworks bugs..?
      return Clamp(carry);
    }
  }

  constexpr PrimeFieldMont& SquareInPlace() {
    if (N == 1) {
      return MulInPlace(*this);
    }

    BigInt<N * 2> r;
    MulResult<uint64_t> mul_result;
    for (size_t i = 0; i < N - 1; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        mul_result = internal::MulAddWithCarry(r[i + j], value_[i], value_[j],
                                               mul_result.hi);
        r[i + j] = mul_result.lo;
      }
      r[i + N] = mul_result.hi;
      mul_result.hi = 0;
    }

    r[2 * N - 1] = r[2 * N - 2] >> 63;
    for (size_t i = 2; i < 2 * N - 1; ++i) {
      r[2 * N - i] = (r[2 * N - i] << 1) | (r[2 * N - (i + 1)] >> 63);
    }
    r[1] <<= 1;

    AddResult<uint64_t> add_result;
    for (size_t i = 0; i < N; ++i) {
      mul_result = internal::MulAddWithCarry(r[2 * i], value_[i], value_[i],
                                             mul_result.hi);
      r[2 * i] = mul_result.lo;
      add_result = internal::AddWithCarry(r[2 * i + 1], 0, mul_result.hi);
      r[2 * i + 1] = add_result.result;
      mul_result.hi = add_result.carry;
    }
    // Montgomery reduction
    add_result.carry = 0;
    for (size_t i = 0; i < N; ++i) {
      uint64_t k = r[i] * kInverse;
      mul_result = internal::MulAddWithCarry(r[i], k, Config::kModulus[0]);
      for (size_t j = 1; j < N; ++j) {
        mul_result = internal::MulAddWithCarry(r[j + i], k, Config::kModulus[j],
                                               mul_result.hi);
        r[j + i] = mul_result.lo;
      }
      add_result =
          internal::AddWithCarry(r[N + i], mul_result.hi, add_result.carry);
    }
    memcpy(value_.limbs, &r.limbs[N], sizeof(uint64_t) * N);
    return Clamp(add_result.carry);
  }

  // MultiplicativeGroup methods
  constexpr PrimeFieldMont& InverseInPlace() {
    CHECK(!IsZero());
    // Guajardo Kumar Paar Pelzl
    // Efficient Software-Implementation of Finite Fields with Applications to
    // Cryptography
    // Algorithm 16 (BEA for Inversion in Fp)

    BigInt<N> u = value_;
    BigInt<N> v = Config::kModulus;
    PrimeFieldMont b;
    b.value_ = kMontgomeryR2;
    PrimeFieldMont c;

    while (!u.IsOne() && !v.IsOne()) {
      while (u.IsEven()) {
        u.DivBy2InPlace();

        if (b.value_.IsEven()) {
          b.value_.DivBy2InPlace();
        } else {
          uint64_t carry = 0;
          b.value_.AddInPlace(Config::kModulus, carry);
          b.value_.DivBy2InPlace();
          if constexpr (!kModulusHasSparseBit) {
            if (carry) {
              b.value_[N - 1] |= static_cast<uint64_t>(1) << 63;
            }
          }
        }
      }

      while (v.IsEven()) {
        v.DivBy2InPlace();

        if (c.value_.IsEven()) {
          c.value_.DivBy2InPlace();
        } else {
          uint64_t carry = 0;
          c.value_.AddInPlace(Config::kModulus, carry);
          c.value_.DivBy2InPlace();
          if constexpr (!kModulusHasSparseBit) {
            if (carry) {
              c.value_[N - 1] |= static_cast<uint64_t>(1) << 63;
            }
          }
        }
      }

      uint64_t unused = 0;
      if (v < u) {
        u.SubInPlace(v, unused);
        b -= c;
      } else {
        v.SubInPlace(u, unused);
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
  constexpr PrimeFieldMont& Clamp(bool carry) {
    bool needs_to_clamp = false;
    if constexpr (kModulusHasSparseBit) {
      needs_to_clamp = value_ >= Config::kModulus;
    } else {
      needs_to_clamp = carry || value_ >= Config::kModulus;
    }
    if (needs_to_clamp) {
      uint64_t unused = 0;
      value_.SubInPlace(Config::kModulus, unused);
    }
    return *this;
  }

  // TODO(chokobole): Support bigendian.
  constexpr PrimeFieldMont& MulWithoutConditionSubtract(
      const PrimeFieldMont& other, uint64_t& carry) {
    BigInt<N> lo;
    BigInt<N> hi;
    MulResult<uint64_t> mul_result;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        size_t k = i + j;
        if (k >= N) {
          mul_result =
              internal::MulAddWithCarry(hi.limbs[k - N], value_.limbs[i],
                                        other.value_.limbs[j], mul_result.hi);
          hi.limbs[k - N] = mul_result.lo;
        } else {
          mul_result =
              internal::MulAddWithCarry(lo.limbs[k], value_.limbs[i],
                                        other.value_.limbs[j], mul_result.hi);
          lo.limbs[k] = mul_result.lo;
        }
      }
    }
    // Montgomery reduction
    AddResult<uint64_t> add_result;
    for (size_t i = 0; i < N; ++i) {
      uint64_t tmp = lo.limbs[i] * kInverse;
      mul_result = internal::MulAddWithCarry(
          lo[i], tmp, Config::kModulus.limbs[0], mul_result.hi);
      for (size_t j = 0; j < N; ++j) {
        size_t k = i + j;
        if (k >= N) {
          mul_result = internal::MulAddWithCarry(
              hi.limbs[k - N], tmp, Config::kModulus.limbs[j], mul_result.hi);
          hi.limbs[k - N] = mul_result.lo;
        } else {
          mul_result = internal::MulAddWithCarry(
              lo.limbs[k], tmp, Config::kModulus.limbs[j], mul_result.hi);
          lo.limbs[k] = mul_result.lo;
        }
      }
      add_result =
          internal::AddWithCarry(hi.limbs[i], add_result.carry, mul_result.hi);
    }

    value_ = hi;
    carry = add_result.carry;
    return *this;
  }

  BigInt<N> value_;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os, const PrimeFieldMont<Config>& f) {
  return os << f.ToString();
}

template <typename Config>
class MultiplicativeIdentity<PrimeFieldMont<Config>> {
 public:
  using F = PrimeFieldMont<Config>;

  static const F& One() {
    static F one(F::One());
    return one;
  }

  constexpr static bool IsOne(const F& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<PrimeFieldMont<Config>> {
 public:
  using F = PrimeFieldMont<Config>;

  static const F& Zero() {
    static F zero(F::Zero());
    return zero;
  }

  constexpr static bool IsZero(const F& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_H_
