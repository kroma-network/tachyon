// Copyright 2024 Ulvetanna Inc.
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.ulvetanna file.

#include <immintrin.h>

#include <algorithm>

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/simd_int.h"

namespace tachyon::math {

namespace {

__m128i ToVector(const SimdInt128& value) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(&value));
}

SimdInt128 FromVector(__m128i vector) {
  SimdInt128 ret;
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&ret), vector);
  return ret;
}

}  // namespace

// static
template <>
SimdInt128 SimdInt128::Broadcast(uint8_t value) {
  return FromVector(_mm_set1_epi8(value));
}

// static
template <>
SimdInt128 SimdInt128::Broadcast(uint16_t value) {
  return FromVector(_mm_set1_epi16(value));
}

// static
template <>
SimdInt128 SimdInt128::Broadcast(uint32_t value) {
  return FromVector(_mm_set1_epi32(value));
}

// static
template <>
SimdInt128 SimdInt128::Broadcast(uint64_t value) {
  return FromVector(_mm_set1_epi64(base::bit_cast<__m64>(value)));
}

template <>
bool SimdInt128::operator==(const SimdInt128& other) const {
  __m128i neq = _mm_xor_si128(ToVector(*this), ToVector(other));
  return _mm_test_all_zeros(neq, neq) == 1;
}

template <>
SimdInt128 SimdInt128::operator&(const SimdInt128& other) const {
  return FromVector(_mm_and_si128(ToVector(*this), ToVector(other)));
}

template <>
SimdInt128 SimdInt128::operator|(const SimdInt128& other) const {
  return FromVector(_mm_or_si128(ToVector(*this), ToVector(other)));
}

template <>
SimdInt128 SimdInt128::operator^(const SimdInt128& other) const {
  return FromVector(_mm_xor_si128(ToVector(*this), ToVector(other)));
}

template <>
SimdInt128 SimdInt128::operator>>(uint32_t count) const {
  if (UNLIKELY(count == 0)) return *this;
  if (UNLIKELY(count >= 128)) return SimdInt128();
  // See
  // https://stackoverflow.com/questions/34478328/the-best-way-to-shift-a-m128i/34482688#34482688
  for (uint32_t i = 1; i < 128; ++i) {
    if (i == count) {
      __m128i carry = _mm_bsrli_si128(ToVector(*this), 8);
      if (count >= 64) {
        return FromVector(
            _mm_srli_epi64(carry, std::max(count - 64, uint32_t{0})));
      } else {
        carry = _mm_slli_epi64(carry, std::max(64 - count, uint32_t{0}));

        __m128i val = _mm_srli_epi64(ToVector(*this), count);
        return FromVector(_mm_or_si128(val, carry));
      }
    }
  }
  NOTREACHED();
  return SimdInt128();
}

template <>
SimdInt128 SimdInt128::operator<<(uint32_t count) const {
  if (UNLIKELY(count == 0)) return *this;
  if (UNLIKELY(count >= 128)) return SimdInt128();
  // See
  // https://stackoverflow.com/questions/34478328/the-best-way-to-shift-a-m128i/34482688#34482688
  for (uint32_t i = 1; i < 128; ++i) {
    if (i == count) {
      __m128i carry = _mm_bslli_si128(ToVector(*this), 8);
      if (count >= 64) {
        return FromVector(
            _mm_slli_epi64(carry, std::max(count - 64, uint32_t{0})));
      } else {
        carry = _mm_srli_epi64(carry, std::max(64 - count, uint32_t{0}));

        __m128i val = _mm_slli_epi64(ToVector(*this), count);
        return FromVector(_mm_or_si128(val, carry));
      }
    }
  }
  NOTREACHED();
  return SimdInt128();
}

}  // namespace tachyon::math
