#ifndef TACHYON_MATH_BASE_BIG_INT_H_
#define TACHYON_MATH_BASE_BIG_INT_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/endian_utils.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/random.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/arithmetics.h"
#include "tachyon/math/base/bit_traits.h"

namespace tachyon::math {
namespace internal {

TACHYON_EXPORT bool StringToLimbs(std::string_view str, uint64_t* limbs,
                                  size_t limb_nums);
TACHYON_EXPORT bool HexStringToLimbs(std::string_view str, uint64_t* limbs,
                                     size_t limb_nums);

TACHYON_EXPORT std::string LimbsToString(const uint64_t* limbs,
                                         size_t limb_nums);
TACHYON_EXPORT std::string LimbsToHexString(const uint64_t* limbs,
                                            size_t limb_nums);

constexpr size_t LimbsAlignment(size_t x) {
  return x % 4 == 0 ? 32 : (x % 2 == 0 ? 16 : 8);
}

}  // namespace internal

template <size_t N>
struct ALIGNAS(internal::LimbsAlignment(N)) BigInt {
  uint64_t limbs[N] = {
      0,
  };
  constexpr static size_t kLimbNums = N;
  constexpr static size_t kSmallestLimbIdx = SMALLEST_INDEX(N);
  constexpr static size_t kBiggestLimbIdx = BIGGEST_INDEX(N);

  constexpr BigInt() = default;
  constexpr explicit BigInt(int value) : BigInt(static_cast<uint64_t>(value)) {
    DCHECK_GE(value, 0);
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr explicit BigInt(T value) {
    limbs[kSmallestLimbIdx] = value;
  }
  constexpr explicit BigInt(std::initializer_list<int> values) {
    DCHECK_LE(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < values.size(); ++i, ++it) {
      DCHECK_GE(*it, 0);
      limbs[i] = *it;
    }
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr explicit BigInt(std::initializer_list<T> values) {
    DCHECK_LE(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < values.size(); ++i, ++it) {
      limbs[i] = *it;
    }
  }
  constexpr explicit BigInt(const uint64_t limbs[N]) {
    memcpy(this->limbs, limbs, sizeof(uint64_t) * N);
  }

  constexpr static BigInt Zero() { return BigInt(0); }

  constexpr static BigInt One() { return BigInt(1); }

  constexpr static BigInt Max() {
    BigInt ret;
    for (uint64_t& limb : ret.limbs) {
      limb = std::numeric_limits<uint64_t>::max();
    }
    return ret;
  }

  constexpr static BigInt Random(const BigInt& max = Max()) {
    BigInt ret;
    for (size_t i = 0; i < N; ++i) {
      ret[i] = base::Uniform<uint64_t, uint64_t>(
          0, std::numeric_limits<uint64_t>::max());
    }
    while (ret >= max) {
      ret.DivBy2InPlace();
    }
    return ret;
  }

  constexpr static BigInt FromDecString(std::string_view str) {
    BigInt ret;
    CHECK(internal::StringToLimbs(str, ret.limbs, N));
    return ret;
  }

  constexpr static BigInt FromHexString(std::string_view str) {
    BigInt ret;
    CHECK(internal::HexStringToLimbs(str, ret.limbs, N));
    return ret;
  }

  constexpr static BigInt FromMontgomery32(const BigInt<N>& value,
                                           const BigInt<N>& modulus,
                                           uint32_t inverse) {
    return FromMontgomery(value, modulus, inverse);
  }

  constexpr static BigInt FromMontgomery64(const BigInt<N>& value,
                                           const BigInt<N>& modulus,
                                           uint64_t inverse) {
    return FromMontgomery(value, modulus, inverse);
  }

  template <size_t N2>
  constexpr BigInt<N2> Extend() const {
    static_assert(N2 > N);
    BigInt<N2> ret;
    for (size_t i = 0; i < N; ++i) {
      ret[i] = limbs[i];
    }
    return ret;
  }

  template <size_t N2>
  constexpr BigInt<N2> Shrink() const {
    static_assert(N2 < N);
    BigInt<N2> ret;
    for (size_t i = 0; i < N2; ++i) {
      ret[i] = limbs[i];
    }
    return ret;
  }

  template <bool ModulusHasSpareBit>
  constexpr static void Clamp(const BigInt& modulus, BigInt* value,
                              [[maybe_unused]] bool carry = false) {
    bool needs_to_clamp = false;
    if constexpr (ModulusHasSpareBit) {
      needs_to_clamp = *value >= modulus;
    } else {
      needs_to_clamp = carry || *value >= modulus;
    }
    if (needs_to_clamp) {
      value->SubInPlace(modulus);
    }
  }

  template <bool ModulusHasSpareBit>
  constexpr static void MontgomeryReduce32(BigInt<2 * N>& r,
                                           const BigInt& modulus,
                                           uint32_t inverse, BigInt* out) {
    MontgomeryReduce<ModulusHasSpareBit>(r, modulus, inverse, out);
  }

  template <bool ModulusHasSpareBit>
  constexpr static void MontgomeryReduce64(BigInt<2 * N>& r,
                                           const BigInt& modulus,
                                           uint64_t inverse, BigInt* out) {
    MontgomeryReduce<ModulusHasSpareBit>(r, modulus, inverse, out);
  }

  constexpr bool IsZero() const {
    for (size_t i = 0; i < N; ++i) {
      if (limbs[i] != 0) return false;
    }
    return true;
  }

  constexpr bool IsOne() const {
    FOR_BUT_SMALLEST(i, N) {
      if (limbs[i] != 0) return false;
    }
    return limbs[kSmallestLimbIdx] == 1;
  }

  constexpr bool IsEven() const { return limbs[kSmallestLimbIdx] % 2 == 0; }
  constexpr bool IsOdd() const { return limbs[kSmallestLimbIdx] % 2 == 1; }

  constexpr uint64_t& biggest_limb() { return limbs[kBiggestLimbIdx]; }
  constexpr const uint64_t& biggest_limb() const {
    return limbs[kBiggestLimbIdx];
  }

  constexpr uint64_t& smallest_limb() { return limbs[kSmallestLimbIdx]; }
  constexpr const uint64_t& smallest_limb() const {
    return limbs[kSmallestLimbIdx];
  }

  constexpr uint64_t ExtractBits64(size_t bit_offset, size_t bit_count) const {
    return ExtractBits<uint64_t>(bit_offset, bit_count);
  }

  constexpr uint32_t ExtractBits32(size_t bit_offset, size_t bit_count) const {
    return ExtractBits<uint32_t>(bit_offset, bit_count);
  }

  constexpr uint64_t& operator[](size_t i) {
    DCHECK_LT(i, N);
    return limbs[i];
  }
  constexpr const uint64_t& operator[](size_t i) const {
    DCHECK_LT(i, N);
    return limbs[i];
  }

  constexpr bool operator==(const BigInt& other) const {
    for (size_t i = 0; i < N; ++i) {
      if (limbs[i] != other.limbs[i]) return false;
    }
    return true;
  }

  constexpr bool operator!=(const BigInt& other) const {
    return !operator==(other);
  }

  constexpr bool operator<(const BigInt& other) const {
    FOR_FROM_BIGGEST(i, 0, N) {
      if (limbs[i] == other.limbs[i]) continue;
      return limbs[i] < other.limbs[i];
    }
    return false;
  }

  constexpr bool operator>(const BigInt& other) const {
    FOR_FROM_BIGGEST(i, 0, N) {
      if (limbs[i] == other.limbs[i]) continue;
      return limbs[i] > other.limbs[i];
    }
    return false;
  }

  constexpr bool operator<=(const BigInt& other) const {
    FOR_FROM_BIGGEST(i, 0, N) {
      if (limbs[i] == other.limbs[i]) continue;
      return limbs[i] < other.limbs[i];
    }
    return true;
  }

  constexpr bool operator>=(const BigInt& other) const {
    FOR_FROM_BIGGEST(i, 0, N) {
      if (limbs[i] == other.limbs[i]) continue;
      return limbs[i] > other.limbs[i];
    }
    return true;
  }

  constexpr BigInt operator+(const BigInt& other) const {
    BigInt ret = *this;
    return ret.AddInPlace(other);
  }

  constexpr BigInt& operator+=(const BigInt& other) {
    return AddInPlace(other);
  }

  constexpr BigInt operator-(const BigInt& other) const {
    BigInt ret = *this;
    return ret.SubInPlace(other);
  }

  constexpr BigInt& operator-=(const BigInt& other) {
    return SubInPlace(other);
  }

  constexpr BigInt operator*(const BigInt& other) const {
    BigInt ret = *this;
    return ret.MulInPlace(other);
  }

  constexpr BigInt& operator*=(const BigInt& other) {
    return MulInPlace(other);
  }

  constexpr BigInt operator/(const BigInt& other) const { return Div(other); }

  constexpr BigInt& operator/=(const BigInt& other) {
    *this = Div(other);
    return *this;
  }

  constexpr BigInt operator%(const BigInt& other) const { return Mod(other); }

  constexpr BigInt& operator%=(const BigInt& other) {
    *this = Mod(other);
    return *this;
  }

  constexpr BigInt& AddInPlace(const BigInt& other) {
    uint64_t unused = 0;
    return AddInPlace(other, unused);
  }

  constexpr BigInt& AddInPlace(const BigInt& other, uint64_t& carry) {
    AddResult<uint64_t> result;

#define ADD_WITH_CARRY_INLINE(num)                                       \
  do {                                                                   \
    if constexpr (N >= (num + 1)) {                                      \
      result = internal::u64::AddWithCarry(limbs[num], other.limbs[num], \
                                           result.carry);                \
      limbs[num] = result.result;                                        \
    }                                                                    \
  } while (false)

#if ARCH_CPU_BIG_ENDIAN
    ADD_WITH_CARRY_INLINE(N - 1);
    ADD_WITH_CARRY_INLINE(N - 2);
    ADD_WITH_CARRY_INLINE(N - 3);
    ADD_WITH_CARRY_INLINE(N - 4);
    ADD_WITH_CARRY_INLINE(N - 5);
    ADD_WITH_CARRY_INLINE(N - 6);
#else  // ARCH_CPU_LITTLE_ENDIAN
    ADD_WITH_CARRY_INLINE(0);
    ADD_WITH_CARRY_INLINE(1);
    ADD_WITH_CARRY_INLINE(2);
    ADD_WITH_CARRY_INLINE(3);
    ADD_WITH_CARRY_INLINE(4);
    ADD_WITH_CARRY_INLINE(5);
#endif

#undef ADD_WITH_CARRY_INLINE

    FOR_FROM_SMALLEST(i, 6, N) {
      result =
          internal::u64::AddWithCarry(limbs[i], other.limbs[i], result.carry);
      limbs[i] = result.result;
    }
    carry = result.carry;
    return *this;
  }

  constexpr BigInt& SubInPlace(const BigInt& other) {
    uint64_t unused = 0;
    return SubInPlace(other, unused);
  }

  constexpr BigInt& SubInPlace(const BigInt& other, uint64_t& borrow) {
    SubResult<uint64_t> result;

#define SUB_WITH_BORROW_INLINE(num)                                       \
  do {                                                                    \
    if constexpr (N >= (num + 1)) {                                       \
      result = internal::u64::SubWithBorrow(limbs[num], other.limbs[num], \
                                            result.borrow);               \
      limbs[num] = result.result;                                         \
    }                                                                     \
  } while (false)

#if ARCH_CPU_BIG_ENDIAN
    SUB_WITH_BORROW_INLINE(N - 1);
    SUB_WITH_BORROW_INLINE(N - 2);
    SUB_WITH_BORROW_INLINE(N - 3);
    SUB_WITH_BORROW_INLINE(N - 4);
    SUB_WITH_BORROW_INLINE(N - 5);
    SUB_WITH_BORROW_INLINE(N - 6);
#else  // ARCH_CPU_LITTLE_ENDIAN
    SUB_WITH_BORROW_INLINE(0);
    SUB_WITH_BORROW_INLINE(1);
    SUB_WITH_BORROW_INLINE(2);
    SUB_WITH_BORROW_INLINE(3);
    SUB_WITH_BORROW_INLINE(4);
    SUB_WITH_BORROW_INLINE(5);
#endif

#undef SUB_WITH_BORROW_INLINE

    FOR_FROM_SMALLEST(i, 6, N) {
      result =
          internal::u64::SubWithBorrow(limbs[i], other.limbs[i], result.borrow);
      limbs[i] = result.result;
    }
    borrow = result.borrow;
    return *this;
  }

  constexpr BigInt& MulBy2InPlace() {
    uint64_t unused = 0;
    return MulBy2InPlace(unused);
  }

  constexpr BigInt& MulBy2InPlace(uint64_t& carry) {
    carry = 0;
    FOR_FROM_SMALLEST(i, 0, N) {
      uint64_t temp = limbs[i] >> 63;
      limbs[i] <<= 1;
      limbs[i] |= carry;
      carry = temp;
    }
    return *this;
  }

  constexpr BigInt& MulBy2ExpInPlace(uint32_t n) {
    if (n >= static_cast<uint32_t>(64 * N)) {
      memset(limbs, 0, sizeof(uint64_t) * N);
      return *this;
    }

    while (n >= 64) {
      uint64_t t = 0;
      FOR_FROM_SMALLEST(i, 0, N) { std::swap(t, limbs[i]); }
      n -= 64;
    }

    if (n > static_cast<uint32_t>(0)) {
      uint64_t t = 0;
      FOR_FROM_SMALLEST(i, 0, N) {
        uint64_t t2 = limbs[i] >> (64 - n);
        limbs[i] <<= n;
        limbs[i] |= t;
        t = t2;
      }
    }
    return *this;
  }

  constexpr BigInt& MulInPlace(const BigInt& other) {
    BigInt hi;
    return MulInPlace(other, hi);
  }

  constexpr BigInt& MulInPlace(const BigInt& other, BigInt& hi) {
    BigInt lo;
    MulResult<uint64_t> mul_result;
    FOR_FROM_SMALLEST(i, 0, N) {
      FOR_FROM_SMALLEST(j, 0, N) {
        uint64_t& limb = (i + j) >= N ? hi.limbs[(i + j) - N] : lo.limbs[i + j];
        mul_result = internal::u64::MulAddWithCarry(
            limb, limbs[i], other.limbs[j], mul_result.hi);
        limb = mul_result.lo;
      }
      hi[i] = mul_result.hi;
      mul_result.hi = 0;
    }
    *this = lo;
    return *this;
  }

  constexpr BigInt<2 * N> Mul(const BigInt& other) {
    BigInt<2 * N> ret;
    BigInt lo = *this;
    BigInt hi;
    lo.MulInPlace(other, hi);
    memcpy(&ret[0], &lo[0], sizeof(uint64_t) * N);
    memcpy(&ret[N], &hi[0], sizeof(uint64_t) * N);
    return ret;
  }

  constexpr BigInt& DivBy2InPlace() {
    uint64_t last = 0;
    FOR_FROM_BIGGEST(i, 0, N) {
      uint64_t temp = limbs[i] << 63;
      limbs[i] >>= 1;
      limbs[i] |= last;
      last = temp;
    }
    return *this;
  }

  constexpr BigInt& DivBy2ExpInPlace(uint32_t n) {
    if (n >= static_cast<uint32_t>(64 * N)) {
      memset(limbs, 0, sizeof(uint64_t) * N);
      return *this;
    }

    if constexpr (N > 1) {
      while (n >= 64) {
        uint64_t t = 0;
        FOR_FROM_BIGGEST(i, 0, N) { std::swap(t, limbs[i]); }
        n -= 64;
      }
    }

    if (n > static_cast<uint32_t>(0)) {
      uint64_t t = 0;
      FOR_FROM_BIGGEST(i, 0, N) {
        uint64_t t2 = limbs[i] << (64 - n);
        limbs[i] >>= n;
        limbs[i] |= t;
        t = t2;
      }
    }
    return *this;
  }

  constexpr BigInt Div(const BigInt& other) const {
    return Divide(other).quotient;
  }

  constexpr BigInt Mod(const BigInt& other) const {
    return Divide(other).remainder;
  }

  constexpr DivResult<BigInt> Divide(const BigInt<N>& divisor) const {
    // Stupid slow base-2 long division taken from
    // https://en.wikipedia.org/wiki/Division_algorithm
    CHECK(!divisor.IsZero());
    BigInt quotient;
    BigInt remainder;
    size_t bits = BitTraits<BigInt>::GetNumBits(*this);
    uint64_t carry = 0;
    uint64_t& smallest_bit = remainder.limbs[kSmallestLimbIdx];
    FOR_FROM_BIGGEST(i, 0, bits) {
      carry = 0;
      remainder.MulBy2InPlace(carry);
      smallest_bit |= BitTraits<BigInt>::TestBit(*this, i);
      if (remainder >= divisor || carry) {
        uint64_t borrow = 0;
        remainder.SubInPlace(divisor, borrow);
        CHECK_EQ(borrow, carry);
        BitTraits<BigInt>::SetBit(quotient, i, 1);
      }
    }
    return {quotient, remainder};
  }

  std::string ToString() const { return internal::LimbsToString(limbs, N); }
  std::string ToHexString() const {
    return internal::LimbsToHexString(limbs, N);
  }

  template <bool ModulusHasSpareBit>
  constexpr BigInt MontgomeryInverse(const BigInt& modulus,
                                     const BigInt& r2) const {
    CHECK(!IsZero());

    // Guajardo Kumar Paar Pelzl
    // Efficient Software-Implementation of Finite Fields with Applications to
    // Cryptography
    // Algorithm 16 (BEA for Inversion in Fp)

    BigInt u = *this;
    BigInt v = modulus;
    BigInt b = r2;
    BigInt c = BigInt::Zero();

    while (!u.IsOne() && !v.IsOne()) {
      while (u.IsEven()) {
        u.DivBy2InPlace();

        if (b.IsEven()) {
          b.DivBy2InPlace();
        } else {
          uint64_t carry = 0;
          b.AddInPlace(modulus, carry);
          b.DivBy2InPlace();
          if constexpr (!ModulusHasSpareBit) {
            if (carry) {
              b[N - 1] |= static_cast<uint64_t>(1) << 63;
            }
          }
        }
      }

      while (v.IsEven()) {
        v.DivBy2InPlace();

        if (c.IsEven()) {
          c.DivBy2InPlace();
        } else {
          uint64_t carry = 0;
          c.AddInPlace(modulus, carry);
          c.DivBy2InPlace();
          if constexpr (!ModulusHasSpareBit) {
            if (carry) {
              c[N - 1] |= static_cast<uint64_t>(1) << 63;
            }
          }
        }
      }

      if (v < u) {
        u.SubInPlace(v);
        if (b >= c) {
          b -= c;
        } else {
          b += (modulus - c);
        }
      } else {
        v.SubInPlace(u);
        if (c >= b) {
          c -= b;
        } else {
          c += (modulus - b);
        }
      }
    }

    if (u.IsOne()) {
      return b;
    } else {
      return c;
    }
  }

 private:
  template <typename T>
  constexpr T ExtractBits(size_t bit_offset, size_t bit_count) const {
    size_t nums = 0;
    size_t bits = 0;
    if constexpr (std::is_same_v<T, uint32_t>) {
      nums = 2 * N;
      bits = 32;
    } else {
      nums = N;
      bits = 64;
    }

    const T* limbs_ptr = reinterpret_cast<const T*>(limbs);
    size_t limb_idx = bit_offset / bits;
    size_t bit_idx = bit_offset % bits;

    T ret;
    if (bit_idx < bits - bit_count || limb_idx == nums - 1) {
      ret = limbs_ptr[limb_idx] >> bit_idx;
    } else {
      ret = (limbs_ptr[limb_idx] >> bit_idx) |
            (limbs_ptr[1 + limb_idx] << (bits - bit_idx));
    }
    T mask = (static_cast<T>(1) << bit_count) - static_cast<T>(1);
    return ret & mask;
  }

  template <typename T>
  constexpr static BigInt FromMontgomery(const BigInt<N>& value,
                                         const BigInt<N>& modulus, T inverse) {
    BigInt<N> r = value;
    T* r_ptr = reinterpret_cast<T*>(r.limbs);
    const T* m_ptr = reinterpret_cast<const T*>(modulus.limbs);
    size_t num = 0;
    MulResult<T> (*mul_add_with_carry)(T, T, T, T);
    if constexpr (std::is_same_v<T, uint32_t>) {
      num = 2 * N;
      mul_add_with_carry = internal::u32::MulAddWithCarry;
    } else {
      num = N;
      mul_add_with_carry = internal::u64::MulAddWithCarry;
    }
    // Montgomery Reduction
    FOR_FROM_SMALLEST(i, 0, num) {
      T k = r_ptr[i] * inverse;
      MulResult<T> result = mul_add_with_carry(r_ptr[i], k, m_ptr[0], 0);
      FOR_FROM_SECOND_SMALLEST(j, 0, num) {
        result =
            mul_add_with_carry(r_ptr[(j + i) % num], k, m_ptr[j], result.hi);
        r_ptr[(j + i) % num] = result.lo;
      }
      r_ptr[i] = result.hi;
    }
    return r;
  }

  template <bool ModulusHasSpareBit, typename T>
  constexpr static void MontgomeryReduce(BigInt<2 * N>& r,
                                         const BigInt& modulus, T inverse,
                                         BigInt* out) {
    T* r_ptr = reinterpret_cast<T*>(r.limbs);
    const T* m_ptr = reinterpret_cast<const T*>(modulus.limbs);
    size_t num = 0;
    MulResult<T> (*mul_add_with_carry)(T, T, T, T);
    AddResult<T> (*add_with_carry)(T, T, T);
    if constexpr (std::is_same_v<T, uint32_t>) {
      num = 2 * N;
      mul_add_with_carry = internal::u32::MulAddWithCarry;
      add_with_carry = internal::u32::AddWithCarry;
    } else {
      num = N;
      mul_add_with_carry = internal::u64::MulAddWithCarry;
      add_with_carry = internal::u64::AddWithCarry;
    }
    AddResult<T> add_result;
    FOR_FROM_SMALLEST(i, 0, num) {
      T tmp = r_ptr[i] * inverse;
      MulResult<T> mul_result;
      mul_result = mul_add_with_carry(r_ptr[i], tmp, m_ptr[0], mul_result.hi);
      FOR_FROM_SECOND_SMALLEST(j, 0, num) {
        mul_result =
            mul_add_with_carry(r_ptr[i + j], tmp, m_ptr[j], mul_result.hi);
        r_ptr[i + j] = mul_result.lo;
      }
      add_result =
          add_with_carry(r_ptr[num + i], mul_result.hi, add_result.carry);
      r_ptr[num + i] = add_result.result;
    }
    memcpy(&(*out)[0], &r[N], sizeof(uint64_t) * N);
    Clamp<ModulusHasSpareBit>(modulus, out, add_result.carry);
  }
};

template <size_t N>
std::ostream& operator<<(std::ostream& os, const BigInt<N>& bigint) {
  return os << bigint.ToString();
}

template <size_t N>
class BitTraits<BigInt<N>> {
 public:
  constexpr static bool kIsDynamic = false;

  constexpr static size_t GetNumBits(const BigInt<N>& _) { return N * 64; }

  constexpr static bool TestBit(const BigInt<N>& bigint, size_t index) {
    size_t limb_index = index >> 6;
    if (limb_index >= N) return false;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = static_cast<uint64_t>(1) << bit_index;
    return (bigint[limb_index] & bit_index_value) == bit_index_value;
  }

  constexpr static void SetBit(BigInt<N>& bigint, size_t index,
                               bool bit_value) {
    size_t limb_index = index >> 6;
    if (limb_index >= N) return;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = static_cast<uint64_t>(1) << bit_index;
    if (bit_value) {
      bigint[limb_index] |= bit_index_value;
    } else {
      bigint[limb_index] &= ~bit_index_value;
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_BIG_INT_H_
