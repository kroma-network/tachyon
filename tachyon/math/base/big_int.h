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

template <size_t LimbNums>
struct ALIGNAS(internal::ComputeAlignment(LimbNums)) BigInt {
  uint64_t limbs[LimbNums] = {
      0,
  };

  constexpr BigInt() = default;
  constexpr explicit BigInt(int value) : BigInt(static_cast<uint64_t>(value)) {
    DCHECK_GE(value, 0);
  }
  constexpr explicit BigInt(uint64_t value) {
#if ARCH_CPU_BIG_ENDIAN
    size_t idx = LimbNums - 1;
#else  // ARCH_CPU_LITTLE_ENDIAN
    size_t idx = 0;
#endif
    limbs[idx] = value;
  }
  constexpr explicit BigInt(std::initializer_list<int> values) {
    DCHECK_EQ(values.size(), LimbNums);
    auto it = values.begin();
    for (size_t i = 0; i < LimbNums; ++i, ++it) {
      DCHECK_GE(*it, 0);
      limbs[i] = *it;
    }
  }
  constexpr explicit BigInt(uint64_t limbs[LimbNums]) : limbs(limbs) {}

  constexpr BigInt(const BigInt& other) {
    memcpy(limbs, other.limbs, sizeof(uint64_t) * LimbNums);
  }
  constexpr BigInt& operator=(const BigInt& other) {
    memcpy(limbs, other.limbs, sizeof(uint64_t) * LimbNums);
    return *this;
  }

  constexpr static BigInt Zero() { return BigInt(0); }

  constexpr static BigInt One() { return BigInt(1); }

  static constexpr BigInt FromDecString(std::string_view str) {
    BigInt ret;
    CHECK(internal::StringToLimbs(str, ret.limbs, LimbNums));
    return ret;
  }

  static constexpr BigInt FromHexString(std::string_view str) {
    BigInt ret;
    CHECK(internal::HexStringToLimbs(str, ret.limbs, LimbNums));
    return ret;
  }

  constexpr bool IsZero() const {
    for (size_t i = 0; i < LimbNums; ++i) {
      if (limbs[i] != 0) return false;
    }
    return true;
  }

  constexpr bool IsOne() const {
#if ARCH_CPU_BIG_ENDIAN
    for (size_t i = 0; i < LimbNums - 1; ++i) {
      if (limbs[i] != 0) return false;
    }
    return limbs[LimbNums - 1] == 1;
#else  // ARCH_CPU_LITTLE_ENDIAN
    for (size_t i = 1; i < LimbNums; ++i) {
      if (limbs[i] != 0) return false;
    }
    return limbs[0] == 1;
#endif
  }

  constexpr uint64_t& operator[](size_t i) {
    DCHECK_LT(i, LimbNums);
    return limbs[i];
  }
  constexpr const uint64_t& operator[](size_t i) const {
    DCHECK_LT(i, LimbNums);
    return limbs[i];
  }

  constexpr bool operator==(const BigInt& other) const {
    for (size_t i = 0; i < LimbNums; ++i) {
      if (limbs[i] != other.limbs[i]) return false;
    }
    return true;
  }

  constexpr bool operator!=(const BigInt& other) const {
    return !operator==(other);
  }

  constexpr bool operator<(const BigInt& other) const {
    for (size_t i = 0; i < LimbNums; ++i) {
#if ARCH_CPU_BIG_ENDIAN
      size_t idx = i;
#else  // ARCH_CPU_LITTLE_ENDIAN
      size_t idx = LimbNums - i - 1;
#endif
      if (limbs[idx] == other.limbs[idx]) continue;
      return limbs[idx] < other.limbs[idx];
    }
    return false;
  }

  constexpr bool operator>(const BigInt& other) const {
    for (size_t i = 0; i < LimbNums; ++i) {
#if ARCH_CPU_BIG_ENDIAN
      size_t idx = i;
#else  // ARCH_CPU_LITTLE_ENDIAN
      size_t idx = LimbNums - i - 1;
#endif
      if (limbs[idx] == other.limbs[idx]) continue;
      return limbs[idx] > other.limbs[idx];
    }
    return false;
  }

  constexpr bool operator<=(const BigInt& other) const {
    for (size_t i = 0; i < LimbNums; ++i) {
#if ARCH_CPU_BIG_ENDIAN
      size_t idx = i;
#else  // ARCH_CPU_LITTLE_ENDIAN
      size_t idx = LimbNums - i - 1;
#endif
      if (limbs[idx] == other.limbs[idx]) continue;
      return limbs[idx] < other.limbs[idx];
    }
    return true;
  }

  constexpr bool operator>=(const BigInt& other) const {
    for (size_t i = 0; i < LimbNums; ++i) {
#if ARCH_CPU_BIG_ENDIAN
      size_t idx = i;
#else  // ARCH_CPU_LITTLE_ENDIAN
      size_t idx = LimbNums - i - 1;
#endif
      if (limbs[idx] == other.limbs[idx]) continue;
      return limbs[idx] > other.limbs[idx];
    }
    return true;
  }

  std::string ToString() const {
    return internal::LimbsToString(limbs, LimbNums);
  }
  std::string ToHexString() const {
    return internal::LimbsToHexString(limbs, LimbNums);
  }
};

template <size_t LimbNums>
std::ostream& operator<<(std::ostream& os, const BigInt<LimbNums>& bigint) {
  return os << bigint.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_BIG_INT_H_
