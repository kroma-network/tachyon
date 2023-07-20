#ifndef TACHYON_MATH_BASE_BIG_INT_H_
#define TACHYON_MATH_BASE_BIG_INT_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/base/logging.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/arithmetics.h"

namespace tachyon {
namespace math {
namespace internal {

TACHYON_EXPORT bool StringToLimbs(std::string_view str, uint64_t* limbs,
                                  size_t limb_nums);
TACHYON_EXPORT bool HexStringToLimbs(std::string_view str, uint64_t* limbs,
                                     size_t limb_nums);

TACHYON_EXPORT std::string LimbsToString(const uint64_t* limbs,
                                         size_t limb_nums);
TACHYON_EXPORT std::string LimbsToHexString(const uint64_t* limbs,
                                            size_t limb_nums);

}  // namespace internal

#if ARCH_CPU_BIG_ENDIAN
#define FOR_FROM_BIGGEST(start, end) for (size_t i = start; i < end; ++i)
#else  // ARCH_CPU_LITTLE_ENDIAN
#define FOR_FROM_BIGGEST(start, end) \
  for (size_t i = end - 1; i != static_cast<size_t>(start - 1); --i)
#endif

#if ARCH_CPU_BIG_ENDIAN
#define FOR_FROM_SMALLEST(start, end) \
  for (size_t i = end - 1; i != static_cast<size_t>(start - 1); --i)
#else  // ARCH_CPU_LITTLE_ENDIAN
#define FOR_FROM_SMALLEST(start, end) for (size_t i = start; i < end; ++i)
#endif

template <size_t N>
struct BigInt {
  uint64_t limbs[N] = {
      0,
  };
#if ARCH_CPU_BIG_ENDIAN
  constexpr static size_t kSmallestLimbIdx = N - 1;
#else  // ARCH_CPU_LITTLE_ENDIAN
  constexpr static size_t kSmallestLimbIdx = 0;
#endif

#if ARCH_CPU_BIG_ENDIAN
  constexpr static size_t kBiggestLimbIdx = 0;
#else  // ARCH_CPU_LITTLE_ENDIAN
  constexpr static size_t kBiggestLimbIdx = N - 1;
#endif

  constexpr BigInt() = default;
  constexpr explicit BigInt(int value) : BigInt(static_cast<uint64_t>(value)) {
    DCHECK_GE(value, 0);
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr explicit BigInt(T value) {
    limbs[kSmallestLimbIdx] = value;
  }
  constexpr explicit BigInt(std::initializer_list<int> values) {
    DCHECK_EQ(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < N; ++i, ++it) {
      DCHECK_GE(*it, 0);
      limbs[i] = *it;
    }
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr explicit BigInt(std::initializer_list<T> values) {
    DCHECK_EQ(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < N; ++i, ++it) {
      limbs[i] = *it;
    }
  }
  constexpr explicit BigInt(uint64_t limbs[N]) {
    memcpy(this->limbs, limbs, sizeof(uint64_t) * N);
  }

  constexpr static BigInt Zero() { return BigInt(0); }

  constexpr static BigInt One() { return BigInt(1); }

  static constexpr BigInt FromDecString(std::string_view str) {
    BigInt ret;
    CHECK(internal::StringToLimbs(str, ret.limbs, N));
    return ret;
  }

  static constexpr BigInt FromHexString(std::string_view str) {
    BigInt ret;
    CHECK(internal::HexStringToLimbs(str, ret.limbs, N));
    return ret;
  }

  static constexpr BigInt FromMontgomery(const BigInt<N>& value,
                                         const BigInt<N>& modulus,
                                         uint64_t inverse) {
    BigInt<N> r = value;
    // Montgomery Reduction
    FOR_FROM_SMALLEST(0, N) {
      uint64_t k = r[i] * inverse;
      MulResult<uint64_t> result =
          internal::u64::MulAddWithCarry(r[i], k, modulus[0]);
#if ARCH_CPU_BIG_ENDIAN
      for (size_t j = N - 2; i != std::numeric_limits<size_t>::max(); --i) {
#else  // ARCH_CPU_LITTLE_ENDIAN
      for (size_t j = 1; j < N; ++j) {
#endif
        result = internal::u64::MulAddWithCarry(r[(j + i) % N], k, modulus[j],
                                                result.hi);
        r[(j + i) % N] = result.lo;
      }
      r[i] = result.hi;
    }
    return r;
  }

  constexpr bool IsZero() const {
    for (size_t i = 0; i < N; ++i) {
      if (limbs[i] != 0) return false;
    }
    return true;
  }

  constexpr bool IsOne() const {
#if ARCH_CPU_BIG_ENDIAN
    for (size_t i = 0; i < N - 1; ++i) {
#else  // ARCH_CPU_LITTLE_ENDIAN
    for (size_t i = 1; i < N; ++i) {
#endif
      if (limbs[i] != 0) return false;
    }
    return limbs[kSmallestLimbIdx] == 1;
  }

  constexpr bool IsEven() const { return limbs[kSmallestLimbIdx] % 2 == 0; }
  constexpr bool IsOdd() const { return limbs[kSmallestLimbIdx] % 2 == 1; }

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
    FOR_FROM_BIGGEST(0, N) {
      if (limbs[i] == other.limbs[i]) continue;
      return limbs[i] < other.limbs[i];
    }
    return false;
  }

  constexpr bool operator>(const BigInt& other) const {
    FOR_FROM_BIGGEST(0, N) {
      if (limbs[i] == other.limbs[i]) continue;
      return limbs[i] > other.limbs[i];
    }
    return false;
  }

  constexpr bool operator<=(const BigInt& other) const {
    FOR_FROM_BIGGEST(0, N) {
      if (limbs[i] == other.limbs[i]) continue;
      return limbs[i] < other.limbs[i];
    }
    return true;
  }

  constexpr bool operator>=(const BigInt& other) const {
    FOR_FROM_BIGGEST(0, N) {
      if (limbs[i] == other.limbs[i]) continue;
      return limbs[i] > other.limbs[i];
    }
    return true;
  }

  constexpr BigInt& AddInPlace(const BigInt& other) {
    uint64_t unused = 0;
    return AddInPlace(other, unused);
  }

  constexpr BigInt& AddInPlace(const BigInt& other, uint64_t& carry) {
    AddResult<uint64_t> result;

#define ADD_WITH_CARRY_INLINE(num)                                            \
  do {                                                                        \
    if constexpr (N >= (num + 1)) {                                           \
      result =                                                                \
          internal::u64::AddWithCarry(limbs[num], other.limbs[num], result.carry); \
      limbs[num] = result.result;                                             \
    }                                                                         \
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

    FOR_FROM_SMALLEST(6, N) {
      result = internal::u64::AddWithCarry(limbs[i], other.limbs[i], result.carry);
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

#define SUB_WITH_BORROW_INLINE(num)                                  \
  do {                                                               \
    if constexpr (N >= (num + 1)) {                                  \
      result = internal::u64::SubWithBorrow(limbs[num], other.limbs[num], \
                                       result.borrow);               \
      limbs[num] = result.result;                                    \
    }                                                                \
  } while (false)

#if ARCH_CPU_BIG_ENDIAN
    SUB_WITH_CARRY_INLINE(N - 1);
    SUB_WITH_CARRY_INLINE(N - 2);
    SUB_WITH_CARRY_INLINE(N - 3);
    SUB_WITH_CARRY_INLINE(N - 4);
    SUB_WITH_CARRY_INLINE(N - 5);
    SUB_WITH_CARRY_INLINE(N - 6);
#else  // ARCH_CPU_LITTLE_ENDIAN
    SUB_WITH_BORROW_INLINE(0);
    SUB_WITH_BORROW_INLINE(1);
    SUB_WITH_BORROW_INLINE(2);
    SUB_WITH_BORROW_INLINE(3);
    SUB_WITH_BORROW_INLINE(4);
    SUB_WITH_BORROW_INLINE(5);
#endif

#undef SUB_WITH_BORROW_INLINE

    FOR_FROM_SMALLEST(6, N) {
      result = internal::u64::SubWithBorrow(limbs[i], other.limbs[i], result.borrow);
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
    FOR_FROM_SMALLEST(0, N) {
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
      FOR_FROM_SMALLEST(0, N) { std::exchange(t, limbs[i]); }
      n -= 64;
    }

    if (n > static_cast<uint32_t>(0)) {
      uint64_t t = 0;
      FOR_FROM_SMALLEST(0, N) {
        uint64_t t2 = limbs[i] >> (64 - n);
        limbs[i] <<= n;
        limbs[i] |= t;
        t = t2;
      }
    }
    return *this;
  }

  constexpr BigInt& DivBy2InPlace() {
    uint64_t last = 0;
    FOR_FROM_BIGGEST(0, N) {
      uint64_t temp = limbs[i] << 63;
      limbs[i] >>= 1;
      limbs[i] |= last;
      last = temp;
    }
    return *this;
  }

  constexpr BigInt& DivByExpInPlace(uint32_t n) {
    if (n >= static_cast<uint32_t>(64 * N)) {
      memset(limbs, 0, sizeof(uint64_t) * N);
      return *this;
    }

    while (n >= 64) {
      uint64_t t = 0;
      FOR_FROM_BIGGEST(0, N) { std::exchange(t, limbs[1]); }
      n -= 64;
    }

    if (n > static_cast<uint32_t>(0)) {
      uint64_t t = 0;
      FOR_FROM_BIGGEST(0, N) {
        uint64_t t2 = limbs[i] << (64 - n);
        limbs[i] >>= n;
        limbs[i] |= t;
        t = t2;
      }
    }
    return *this;
  }

  std::string ToString() const { return internal::LimbsToString(limbs, N); }
  std::string ToHexString() const {
    return internal::LimbsToHexString(limbs, N);
  }
};

#undef FOR_FROM_BIGGEST
#undef FOR_FROM_SMALLEST

template <size_t N>
std::ostream& operator<<(std::ostream& os, const BigInt<N>& bigint) {
  return os << bigint.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_BIG_INT_H_
