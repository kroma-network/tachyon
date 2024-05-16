// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_BASE_BIG_INT_H_
#define TACHYON_MATH_BASE_BIG_INT_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <bitset>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/endian_utils.h"
#include "tachyon/base/json/json.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/random.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/arithmetics.h"
#include "tachyon/math/base/bit_traits_forward.h"

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
                                            size_t limb_nums, bool pad_zero);

constexpr size_t LimbsAlignment(size_t x) {
  return x % 4 == 0 ? 32 : (x % 2 == 0 ? 16 : 8);
}

}  // namespace internal

// BigInt is a fixed size array of uint64_t, capable of holding up to |N| limbs,
// designed to support a wide range of big integer arithmetic operations.
template <size_t N>
struct ALIGNAS(internal::LimbsAlignment(N)) BigInt {
  uint64_t limbs[N] = {
      0,
  };
  constexpr static size_t kLimbNums = N;
  constexpr static size_t kSmallestLimbIdx = SMALLEST_INDEX(N);
  constexpr static size_t kBiggestLimbIdx = BIGGEST_INDEX(N);
  constexpr static size_t kLimbByteNums = sizeof(uint64_t);
  constexpr static size_t kByteNums = N * sizeof(uint64_t);
  constexpr static size_t kLimbBitNums = kLimbByteNums * 8;
  constexpr static size_t kBitNums = kByteNums * 8;

  constexpr BigInt() = default;
  constexpr explicit BigInt(int64_t value)
      : BigInt(static_cast<uint64_t>(value)) {
    DCHECK_GE(value, int64_t{0});
  }
  constexpr explicit BigInt(uint64_t value) { limbs[kSmallestLimbIdx] = value; }
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

  // Returns the maximum representable value for BigInt.
  constexpr static BigInt Max() {
    BigInt ret;
    for (uint64_t& limb : ret.limbs) {
      limb = std::numeric_limits<uint64_t>::max();
    }
    return ret;
  }

  // Generate a random BigInt between [0, |max|).
  constexpr static BigInt Random(const BigInt& max = Max()) {
    BigInt ret;
    for (size_t i = 0; i < N; ++i) {
      ret[i] = base::Uniform(base::Range<uint64_t>::All());
    }
    while (ret >= max) {
      ret.DivBy2InPlace();
    }
    return ret;
  }

  // Convert a decimal string to a BigInt.
  constexpr static std::optional<BigInt> FromDecString(std::string_view str) {
    BigInt ret;
    if (!internal::StringToLimbs(str, ret.limbs, N)) return std::nullopt;
    return ret;
  }

  // Convert a hexadecimal string to a BigInt.
  constexpr static std::optional<BigInt> FromHexString(std::string_view str) {
    BigInt ret;
    if (!(internal::HexStringToLimbs(str, ret.limbs, N))) return std::nullopt;
    return ret;
  }

  // Constructs a BigInt value from a given array of bits in little-endian
  // order.
  template <size_t BitNums = kBitNums>
  constexpr static BigInt FromBitsLE(const std::bitset<BitNums>& bits) {
    static_assert(BitNums <= kBitNums);
    BigInt ret;
    size_t bit_idx = 0;
    size_t limb_idx = 0;
    std::bitset<kLimbBitNums> limb_bits;
    FOR_FROM_SMALLEST(i, 0, BitNums) {
      limb_bits.set(bit_idx++, bits[i]);
      bool set = bit_idx == kLimbBitNums;
#if ARCH_CPU_BIG_ENDIAN
      set |= (i == 0);
#else
      set |= (i == BitNums - 1);
#endif
      if (set) {
        uint64_t limb = base::bit_cast<uint64_t>(limb_bits.to_ullong());
        ret.limbs[limb_idx++] = limb;
        limb_bits.reset();
        bit_idx = 0;
      }
    }
    return ret;
  }

  // Constructs a BigInt value from a given array of bits in big-endian order.
  template <size_t BitNums = kBitNums>
  constexpr static BigInt FromBitsBE(const std::bitset<BitNums>& bits) {
    static_assert(BitNums <= kBitNums);
    BigInt ret;
    std::bitset<kLimbBitNums> limb_bits;
    size_t bit_idx = 0;
    size_t limb_idx = 0;
    FOR_FROM_BIGGEST(i, 0, BitNums) {
      limb_bits.set(bit_idx++, bits[i]);
      bool set = bit_idx == kLimbBitNums;
#if ARCH_CPU_BIG_ENDIAN
      set |= (i == BitNums - 1);
#else
      set |= (i == 0);
#endif
      if (set) {
        uint64_t limb = base::bit_cast<uint64_t>(limb_bits.to_ullong());
        ret.limbs[limb_idx++] = limb;
        limb_bits.reset();
        bit_idx = 0;
      }
    }
    return ret;
  }

  // Constructs a BigInt value from a given byte container interpreted in
  // little-endian order. The method processes each byte of the input, packs
  // them into 64-bit limbs, and then sets these limbs in the resulting BigInt.
  // If the system is big-endian, adjustments are made to ensure correct byte
  // ordering.
  template <typename ByteContainer>
  constexpr static BigInt FromBytesLE(const ByteContainer& bytes) {
    BigInt ret;
    size_t byte_idx = 0;
    size_t limb_idx = 0;
    uint64_t limb = 0;
    FOR_FROM_SMALLEST(i, 0, std::size(bytes)) {
      reinterpret_cast<uint8_t*>(&limb)[byte_idx++] = bytes[i];
      bool set = byte_idx == kLimbByteNums;
#if ARCH_CPU_BIG_ENDIAN
      set |= (i == 0);
#else
      set |= (i == std::size(bytes) - 1);
#endif
      if (set) {
        ret.limbs[limb_idx++] = limb;
        limb = 0;
        byte_idx = 0;
      }
    }
    return ret;
  }

  // Constructs a BigInt value from a given byte container interpreted in
  // big-endian order. The method processes each byte of the input, packs them
  // into 64-bit limbs, and then sets these limbs in the resulting BigInt. If
  // the system is little-endian, adjustments are made to ensure correct byte
  // ordering.
  template <typename ByteContainer>
  constexpr static BigInt FromBytesBE(const ByteContainer& bytes) {
    BigInt ret;
    size_t byte_idx = 0;
    size_t limb_idx = 0;
    uint64_t limb = 0;
    FOR_FROM_BIGGEST(i, 0, std::size(bytes)) {
      reinterpret_cast<uint8_t*>(&limb)[byte_idx++] = bytes[i];
      bool set = byte_idx == kLimbByteNums;
#if ARCH_CPU_BIG_ENDIAN
      set |= (i == std::size(bytes) - 1);
#else
      set |= (i == 0);
#endif
      if (set) {
        ret.limbs[limb_idx++] = limb;
        limb = 0;
        byte_idx = 0;
      }
    }
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

  // Extend the current |N| size BigInt to a larger |N2| size.
  template <size_t N2>
  constexpr BigInt<N2> Extend() const {
    static_assert(N2 > N);
    BigInt<N2> ret;
    for (size_t i = 0; i < N; ++i) {
      ret[i] = limbs[i];
    }
    return ret;
  }

  // Shrink the current |N| size BigInt to a smaller |N2| size.
  template <size_t N2>
  constexpr BigInt<N2> Shrink() const {
    static_assert(N2 < N);
    BigInt<N2> ret;
    for (size_t i = 0; i < N2; ++i) {
      ret[i] = limbs[i];
    }
    return ret;
  }

  // Clamp the BigInt value with respect to a modulus.
  // If the value is larger than or equal to the modulus, then the modulus is
  // subtracted from the value. The function considers a spare bit in the
  // modulus based on the template parameter.
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

  // Return the largest (most significant) limb of the BigInt.
  constexpr uint64_t& biggest_limb() { return limbs[kBiggestLimbIdx]; }
  constexpr const uint64_t& biggest_limb() const {
    return limbs[kBiggestLimbIdx];
  }

  // Return the smallest (least significant) limb of the BigInt.
  constexpr uint64_t& smallest_limb() { return limbs[kSmallestLimbIdx]; }
  constexpr const uint64_t& smallest_limb() const {
    return limbs[kSmallestLimbIdx];
  }

  // Extracts a specified number of bits starting from a given bit offset and
  // returns them as a uint64_t.
  constexpr uint64_t ExtractBits64(size_t bit_offset, size_t bit_count) const {
    return ExtractBits<uint64_t>(bit_offset, bit_count);
  }

  // Extracts a specified number of bits starting from a given bit offset and
  // returns them as a uint32_t.
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

  constexpr BigInt operator+(const BigInt& other) const { return Add(other); }

  constexpr BigInt& operator+=(const BigInt& other) {
    return AddInPlace(other);
  }

  constexpr BigInt operator-(const BigInt& other) const { return Sub(other); }

  constexpr BigInt& operator-=(const BigInt& other) {
    return SubInPlace(other);
  }

  constexpr BigInt operator*(const BigInt& other) const { return Mul(other); }

  constexpr BigInt& operator*=(const BigInt& other) {
    *this = Mul(other);
    return *this;
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

  constexpr BigInt Add(const BigInt& other) const {
    uint64_t unused = 0;
    return Add(other, unused);
  }

  constexpr BigInt Add(const BigInt& other, uint64_t& carry) const {
    BigInt ret;
    DoAdd(*this, other, carry, ret);
    return ret;
  }

  constexpr BigInt& AddInPlace(const BigInt& other) {
    uint64_t unused = 0;
    return AddInPlace(other, unused);
  }

  constexpr BigInt& AddInPlace(const BigInt& other, uint64_t& carry) {
    DoAdd(*this, other, carry, *this);
    return *this;
  }

  constexpr BigInt Sub(const BigInt& other) const {
    uint64_t unused = 0;
    return Sub(other, unused);
  }

  constexpr BigInt Sub(const BigInt& other, uint64_t& borrow) const {
    BigInt ret;
    DoSub(*this, other, borrow, ret);
    return ret;
  }

  constexpr BigInt& SubInPlace(const BigInt& other) {
    uint64_t unused = 0;
    return SubInPlace(other, unused);
  }

  constexpr BigInt& SubInPlace(const BigInt& other, uint64_t& borrow) {
    DoSub(*this, other, borrow, *this);
    return *this;
  }

  constexpr BigInt MulBy2() const {
    uint64_t unused = 0;
    return MulBy2(unused);
  }

  constexpr BigInt MulBy2(uint64_t& carry) const {
    BigInt ret;
    DoMulBy2(*this, carry, ret);
    return ret;
  }

  constexpr BigInt& MulBy2InPlace() {
    uint64_t unused = 0;
    return MulBy2InPlace(unused);
  }

  constexpr BigInt& MulBy2InPlace(uint64_t& carry) {
    DoMulBy2(*this, carry, *this);
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

    if (n > uint32_t{0}) {
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

  constexpr BigInt Mul(const BigInt& other) const { return Multiply(other).lo; }

  constexpr BigInt<2 * N> MulExtend(const BigInt& other) const {
    MulResult<BigInt> result = Multiply(other);
    BigInt<2 * N> ret;
    memcpy(&ret[0], &result.lo[0], sizeof(uint64_t) * N);
    memcpy(&ret[N], &result.hi[0], sizeof(uint64_t) * N);
    return ret;
  }

  constexpr MulResult<BigInt> Multiply(const BigInt& other) const {
    MulResult<BigInt> ret;
    MulResult<uint64_t> mul_result;
    FOR_FROM_SMALLEST(i, 0, N) {
      FOR_FROM_SMALLEST(j, 0, N) {
        uint64_t& limb = (i + j) >= N ? ret.hi[(i + j) - N] : ret.lo[i + j];
        mul_result = internal::u64::MulAddWithCarry(limb, limbs[i], other[j],
                                                    mul_result.hi);
        limb = mul_result.lo;
      }
      ret.hi[i] = mul_result.hi;
      mul_result.hi = 0;
    }
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

    if (n > uint32_t{0}) {
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
  std::string ToHexString(bool pad_zero = false) const {
    return internal::LimbsToHexString(limbs, N, pad_zero);
  }

  // Converts the BigInt to a bit array in little-endian.
  template <size_t BitNums = kBitNums>
  std::bitset<BitNums> ToBitsLE() const {
    std::bitset<BitNums> ret;
    size_t bit_w_idx = 0;
    FOR_FROM_SMALLEST(i, 0, BitNums) {
      size_t limb_idx = i / kLimbBitNums;
      size_t bit_r_idx = i % kLimbBitNums;
      bool bit = (limbs[limb_idx] & (uint64_t{1} << bit_r_idx)) >> bit_r_idx;
      ret.set(bit_w_idx++, bit);
    }
    return ret;
  }

  // Converts the BigInt to a bit array in big-endian.
  template <size_t BitNums = kBitNums>
  std::bitset<BitNums> ToBitsBE() const {
    std::bitset<BitNums> ret;
    size_t bit_w_idx = 0;
    FOR_FROM_BIGGEST(i, 0, BitNums) {
      size_t limb_idx = i / kLimbBitNums;
      size_t bit_r_idx = i % kLimbBitNums;
      bool bit = (limbs[limb_idx] & (uint64_t{1} << bit_r_idx)) >> bit_r_idx;
      ret.set(bit_w_idx++, bit);
    }
    return ret;
  }

  // Converts the BigInt to a byte array in little-endian order. This method
  // processes the limbs of the BigInt, extracts individual bytes, and sets them
  // in the resulting array.
  std::array<uint8_t, kByteNums> ToBytesLE() const {
    std::array<uint8_t, kByteNums> ret;
    auto it = ret.begin();
    FOR_FROM_SMALLEST(i, 0, kByteNums) {
      size_t limb_idx = i / kLimbByteNums;
      uint64_t limb = limbs[limb_idx];
      size_t byte_r_idx = i % kLimbByteNums;
      *(it++) = reinterpret_cast<uint8_t*>(&limb)[byte_r_idx];
    }
    return ret;
  }

  // Converts the BigInt to a byte array in big-endian order. This method
  // processes the limbs of the BigInt, extracts individual bytes, and sets them
  // in the resulting array.
  std::array<uint8_t, kByteNums> ToBytesBE() const {
    std::array<uint8_t, kByteNums> ret;
    auto it = ret.begin();
    FOR_FROM_BIGGEST(i, 0, kByteNums) {
      size_t limb_idx = i / kLimbByteNums;
      uint64_t limb = limbs[limb_idx];
      size_t byte_r_idx = i % kLimbByteNums;
      *(it++) = reinterpret_cast<uint8_t*>(&limb)[byte_r_idx];
    }
    return ret;
  }

  template <bool ModulusHasSpareBit>
  constexpr BigInt MontgomeryInverse(const BigInt& modulus,
                                     const BigInt& r2) const {
    // See https://github.com/kroma-network/tachyon/issues/76
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
              b[N - 1] |= uint64_t{1} << 63;
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
              c[N - 1] |= uint64_t{1} << 63;
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

  // TODO(chokobole): This can be optimized since the element of vector occupies
  // fixed 2 bits, we can save much space. e.g, in a worst case for 4 limbs(254
  // bits), 254 * 2 / 8 = 63.4 < 8 * sizeof(uint64_t).
  //
  // This converts bigint to NAF(Non-Adjacent-Form).
  // e.g, 7 = (1 1 1)₂ = (1 0 0 -1)₂
  // See https://en.wikipedia.org/wiki/Non-adjacent_form
  // See cyclotomic_multiplicative_subgroup.h for use case.
  std::vector<int8_t> ToNAF() const {
    BigInt v(*this);
    std::vector<int8_t> ret;
    ret.reserve(8 * sizeof(uint64_t) * N);
    while (!v.IsZero()) {
      int8_t z;
      // v = v₀ * 2⁰ + v₁ * 2¹ + v₂ * 2² + ... + vₙ₋₁ * 2ⁿ⁻¹
      // if v₀ == 0:
      //   z = 0
      //   v = z * 2⁰ + v₁ * 2¹ + v₂ * 2² + ... + vₙ₋₁ * 2ⁿ⁻¹
      // else if v₀ == 1 && v₁ == 0:
      //   z = 2 - 1 = 1
      //   v = z * 2⁰ + v₂ * 2² + ... + vₙ₋₁ * 2ⁿ⁻¹
      // else if v₀ == 1 && v₁ == 1:
      //   z = 2 - 3 = -1
      //   v = z * 2⁰ + (v₂ + 1) * 2² + ... + vₙ₋₁ * 2ⁿ⁻¹
      if (v.IsOdd()) {
        z = 2 - (v[kSmallestLimbIdx] % 4);
        if (z >= 0) {
          v -= BigInt(z);
        } else {
          v += BigInt(-z);
        }
      } else {
        z = 0;
      }
      ret.push_back(z);
      v.DivBy2InPlace();
    }
    return ret;
  }

 private:
  constexpr static void DoAdd(const BigInt& a, const BigInt& b, uint64_t& carry,
                              BigInt& c) {
    AddResult<uint64_t> result;

#define ADD_WITH_CARRY_INLINE(num)                                        \
  do {                                                                    \
    if constexpr (N >= (num + 1)) {                                       \
      result = internal::u64::AddWithCarry(a[num], b[num], result.carry); \
      c[num] = result.result;                                             \
    }                                                                     \
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
      result = internal::u64::AddWithCarry(a[i], b[i], result.carry);
      c[i] = result.result;
    }
    carry = result.carry;
  }

  constexpr static void DoSub(const BigInt& a, const BigInt& b,
                              uint64_t& borrow, BigInt& c) {
    SubResult<uint64_t> result;

#define SUB_WITH_BORROW_INLINE(num)                                         \
  do {                                                                      \
    if constexpr (N >= (num + 1)) {                                         \
      result = internal::u64::SubWithBorrow(a[num], b[num], result.borrow); \
      c[num] = result.result;                                               \
    }                                                                       \
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
      result = internal::u64::SubWithBorrow(a[i], b[i], result.borrow);
      c[i] = result.result;
    }
    borrow = result.borrow;
  }

  constexpr static void DoMulBy2(const BigInt& a, uint64_t& carry, BigInt& b) {
    FOR_FROM_SMALLEST(i, 0, N) {
      uint64_t temp = a[i] >> 63;
      b[i] = a[i] << 1;
      b[i] |= carry;
      carry = temp;
    }
  }

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
    T mask = (T{1} << bit_count) - T{1};
    return ret & mask;
  }

  // Montgomery arithmetic is a technique that allows modular arithmetic to be
  // done more efficiently, by avoiding the need for explicit divisions.
  // See https://en.wikipedia.org/wiki/Montgomery_modular_multiplication

  // Converts a BigInt value from the Montgomery domain back to the standard
  // domain. |FromMontgomery()| performs the Montgomery reduction algorithm to
  // transform a value from the Montgomery domain back to its standard
  // representation.
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

  // Performs Montgomery reduction on a doubled-sized BigInt, and populates
  // |out| with the result.

  // Inputs:
  // - r: A BigInt representing a value (typically A x B) in Montgomery form.
  // - modulus: The modulus M against which we're performing arithmetic.
  // - inverse: The multiplicative inverse of the radix w.r.t. the modulus.

  // Operation:
  // 1. For each limb of r:
  //    - Compute a tmp = r(current limb) * inverse.
  //      This value aids in eliminating the lowest limb of r when multiplied by
  //      the modulus.
  //    - Incrementally add tmp * (modulus to r), effectively canceling out its
  //      current lowest limb.
  //
  // 2. After iterating over all limbs, the higher half of r is the
  //    Montgomery-reduced result of the original operation (like A x B). This
  //    result remains in the Montgomery domain.
  //
  // 3. Apply a final correction (if necessary) to ensure the result is less
  //    than |modulus|.
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
class BitTraits<BigInt<N>> {
 public:
  constexpr static bool kIsDynamic = false;

  constexpr static size_t GetNumBits(const BigInt<N>& _) { return N * 64; }

  constexpr static bool TestBit(const BigInt<N>& bigint, size_t index) {
    size_t limb_index = index >> 6;
    if (limb_index >= N) return false;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = uint64_t{1} << bit_index;
    return (bigint[limb_index] & bit_index_value) == bit_index_value;
  }
  constexpr static void SetBit(BigInt<N>& bigint, size_t index,
                               bool bit_value) {
    size_t limb_index = index >> 6;
    if (limb_index >= N) return;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = uint64_t{1} << bit_index;
    if (bit_value) {
      bigint[limb_index] |= bit_index_value;
    } else {
      bigint[limb_index] &= ~bit_index_value;
    }
  }
};

}  // namespace math

namespace base {

template <size_t N>
class Copyable<math::BigInt<N>> {
 public:
  static bool WriteTo(const math::BigInt<N>& bigint, Buffer* buffer) {
    return buffer->Write(bigint.limbs);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, math::BigInt<N>* bigint) {
    return buffer.Read(bigint->limbs);
  }

  static size_t EstimateSize(const math::BigInt<N>& bigint) {
    return base::EstimateSize(bigint.limbs);
  }
};

template <size_t N>
class RapidJsonValueConverter<math::BigInt<N>> {
 public:
  // NOTE(dongchangYoo): Classes inheriting |BigInt<N>| have a member of
  // type |uint64_t[N]|, but for the sake of readability in the JSON file,
  // |BigInt<N>| is implemented to be converted into a hexadecimal string.
  template <typename Allocator>
  static rapidjson::Value From(const math::BigInt<N>& value,
                               Allocator& allocator) {
    std::string value_str = value.ToHexString();
    return RapidJsonValueConverter<std::string>::From(value_str, allocator);
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::BigInt<N>* value, std::string* error) {
    if (!json_value.IsString()) {
      *error = RapidJsonMismatchedTypeError(key, "string", json_value);
      return false;
    }
    std::string_view value_str;
    if (!RapidJsonValueConverter<std::string_view>::To(json_value, "",
                                                       &value_str, error))
      return false;
    std::optional<math::BigInt<N>> tmp =
        math::BigInt<N>::FromHexString(value_str);
    if (!tmp.has_value()) return false;
    *value = std::move(tmp).value();
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_BIG_INT_H_
