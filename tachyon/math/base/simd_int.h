// Copyright 2024 Ulvetanna Inc.
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.ulvetanna file.

#ifndef TACHYON_MATH_BASE_SIMD_INT_H_
#define TACHYON_MATH_BASE_SIMD_INT_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <type_traits>

#include "tachyon/build/build_config.h"
#include "tachyon/export.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon::math {

template <size_t Bits>
class SimdInt {
 public:
  constexpr static size_t kBits = Bits;
  constexpr static size_t kLimbNums = Bits / 64;

  using value_type = BigInt<kLimbNums>;

  SimdInt() = default;
  template <typename T,
            std::enable_if_t<std::is_constructible_v<value_type, T>>* = nullptr>
  explicit SimdInt(T value) : SimdInt(value_type(value)) {}
  explicit SimdInt(const value_type& value) : value_(value) {}

  static SimdInt Zero() { return SimdInt(); }
  static SimdInt One() { return SimdInt(1); }
  static SimdInt Max() { return SimdInt(value_type::Max()); }
  static SimdInt Broadcast(uint8_t value);
  static SimdInt Broadcast(uint16_t value);
  static SimdInt Broadcast(uint32_t value);
  static SimdInt Broadcast(uint64_t value);
  static SimdInt Random() { return SimdInt(value_type::Random()); }

  const value_type& value() const { return value_; }

  bool IsZero() const { return *this == Zero(); }
  bool IsOne() const { return *this == One(); }
  bool IsMax() const { return *this == Max(); }

  bool operator==(const SimdInt& other) const;
  bool operator!=(const SimdInt& other) const { return !operator==(other); }

  SimdInt operator&(const SimdInt& other) const;
  SimdInt& operator&=(const SimdInt& other) { return *this = *this & other; }

  SimdInt operator|(const SimdInt& other) const;
  SimdInt& operator|=(const SimdInt& other) { return *this = *this | other; }

  SimdInt operator^(const SimdInt& other) const;
  SimdInt& operator^=(const SimdInt& other) { return *this = *this ^ other; }

  SimdInt operator!() const { return *this ^ Max(); }

  SimdInt operator>>(uint32_t count) const;
  SimdInt& operator>>=(uint32_t count) { return *this = *this >> count; }

  SimdInt operator<<(uint32_t count) const;
  SimdInt& operator<<=(uint32_t count) { return *this = *this << count; }

  std::string ToString() const { return value_.ToString(); }
  std::string ToHexString(bool pad_zero = false) const {
    return value_.ToHexString(pad_zero);
  }

 private:
  value_type value_;
};

// clang-format off
#define SPECIALIZE_SIMD_INT(bits)                                           \
  using SimdInt##bits = SimdInt<bits>;                                      \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::Broadcast(uint8_t value);                    \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::Broadcast(uint16_t value);                   \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::Broadcast(uint32_t value);                   \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::Broadcast(uint64_t value);                   \
                                                                            \
  template <>                                                               \
  bool SimdInt##bits::operator==(const SimdInt##bits& value) const;         \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::operator&(const SimdInt##bits& value) const; \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::operator|(const SimdInt##bits& value) const; \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::operator^(const SimdInt##bits& value) const; \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::operator>>(uint32_t count) const;            \
                                                                            \
  template <>                                                               \
  SimdInt##bits SimdInt##bits::operator<<(uint32_t count) const
// clang-format on

SPECIALIZE_SIMD_INT(128);
#if ARCH_CPU_X86_64
SPECIALIZE_SIMD_INT(256);
#endif

#undef SPECIALIZE_SIMD_INT

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_SIMD_INT_H_
