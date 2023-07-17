#ifndef TACHYON_MATH_BASE_BIG_INT_H_
#define TACHYON_MATH_BASE_BIG_INT_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/logging.h"
#include "tachyon/build/build_config.h"

namespace tachyon {
namespace math {
namespace internal {

constexpr size_t ComputeAlignment(size_t x) { return x % 4 == 0 ? 16 : 8; }

TACHYON_EXPORT bool StringToLimbs(std::string_view str, uint64_t* limbs,
                                  size_t limb_nums);
TACHYON_EXPORT bool HexStringToLimbs(std::string_view str, uint64_t* limbs,
                                     size_t limb_nums);

TACHYON_EXPORT std::string LimbsToString(const uint64_t* limbs,
                                         size_t limb_nums);
TACHYON_EXPORT std::string LimbsToHexString(const uint64_t* limbs,
                                            size_t limb_nums);

}  // namespace internal

template <size_t N>
struct ALIGNAS(internal::ComputeAlignment(N)) BigInt {
  uint64_t limbs[N] = {
      0,
  };

  constexpr BigInt() = default;
  constexpr explicit BigInt(int value) : BigInt(static_cast<uint64_t>(value)) {
    DCHECK_GE(value, 0);
  }
  constexpr explicit BigInt(uint64_t value) {
#if ARCH_CPU_BIG_ENDIAN
    size_t idx = N - 1;
#else  // ARCH_CPU_LITTLE_ENDIAN
    size_t idx = 0;
#endif
    limbs[idx] = value;
  }
  constexpr explicit BigInt(std::initializer_list<int> values) {
    DCHECK_EQ(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < N; ++i, ++it) {
      DCHECK_GE(*it, 0);
      limbs[i] = *it;
    }
  }
  constexpr explicit BigInt(uint64_t limbs[N]) : limbs(limbs) {}

  constexpr BigInt(const BigInt& other) {
    memcpy(limbs, other.limbs, sizeof(uint64_t) * N);
  }
  constexpr BigInt& operator=(const BigInt& other) {
    memcpy(limbs, other.limbs, sizeof(uint64_t) * N);
    return *this;
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

  constexpr bool IsZero() const {
    for (size_t i = 0; i < N; ++i) {
      if (limbs[i] != 0) return false;
    }
    return true;
  }

  constexpr bool IsOne() const {
#if ARCH_CPU_BIG_ENDIAN
    for (size_t i = 0; i < N - 1; ++i) {
      if (limbs[i] != 0) return false;
    }
    return limbs[N - 1] == 1;
#else  // ARCH_CPU_LITTLE_ENDIAN
    for (size_t i = 1; i < N; ++i) {
      if (limbs[i] != 0) return false;
    }
    return limbs[0] == 1;
#endif
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
    for (size_t i = 0; i < N; ++i) {
#if ARCH_CPU_BIG_ENDIAN
      size_t idx = i;
#else  // ARCH_CPU_LITTLE_ENDIAN
      size_t idx = N - i - 1;
#endif
      if (limbs[idx] == other.limbs[idx]) continue;
      return limbs[idx] < other.limbs[idx];
    }
    return false;
  }

  constexpr bool operator>(const BigInt& other) const {
    for (size_t i = 0; i < N; ++i) {
#if ARCH_CPU_BIG_ENDIAN
      size_t idx = i;
#else  // ARCH_CPU_LITTLE_ENDIAN
      size_t idx = N - i - 1;
#endif
      if (limbs[idx] == other.limbs[idx]) continue;
      return limbs[idx] > other.limbs[idx];
    }
    return false;
  }

  constexpr bool operator<=(const BigInt& other) const {
    for (size_t i = 0; i < N; ++i) {
#if ARCH_CPU_BIG_ENDIAN
      size_t idx = i;
#else  // ARCH_CPU_LITTLE_ENDIAN
      size_t idx = N - i - 1;
#endif
      if (limbs[idx] == other.limbs[idx]) continue;
      return limbs[idx] < other.limbs[idx];
    }
    return true;
  }

  constexpr bool operator>=(const BigInt& other) const {
    for (size_t i = 0; i < N; ++i) {
#if ARCH_CPU_BIG_ENDIAN
      size_t idx = i;
#else  // ARCH_CPU_LITTLE_ENDIAN
      size_t idx = N - i - 1;
#endif
      if (limbs[idx] == other.limbs[idx]) continue;
      return limbs[idx] > other.limbs[idx];
    }
    return true;
  }

  std::string ToString() const { return internal::LimbsToString(limbs, N); }
  std::string ToHexString() const {
    return internal::LimbsToHexString(limbs, N);
  }
};

template <size_t N>
std::ostream& operator<<(std::ostream& os, const BigInt<N>& bigint) {
  return os << bigint.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_BIG_INT_H_
