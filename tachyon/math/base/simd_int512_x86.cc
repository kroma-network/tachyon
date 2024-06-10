// Copyright 2024 Ulvetanna Inc.
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.ulvetanna file.

#include <immintrin.h>

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/compiler_specific.h"
#include "tachyon/math/base/simd_int.h"

namespace tachyon::math {

namespace {

__m512i ToVector(const SimdInt512& value) { return _mm512_loadu_si512(&value); }

SimdInt512 FromVector(__m512i vector) {
  SimdInt512 ret;
  _mm512_storeu_si512(&ret, vector);
  return ret;
}

}  // namespace

// static
template <>
SimdInt512 SimdInt512::Broadcast(uint8_t value) {
  return FromVector(_mm512_set1_epi8(value));
}

// static
template <>
SimdInt512 SimdInt512::Broadcast(uint16_t value) {
  return FromVector(_mm512_set1_epi16(value));
}

// static
template <>
SimdInt512 SimdInt512::Broadcast(uint32_t value) {
  return FromVector(_mm512_set1_epi32(value));
}

// static
template <>
SimdInt512 SimdInt512::Broadcast(uint64_t value) {
  return FromVector(_mm512_set1_epi64(value));
}

template <>
bool SimdInt512::operator==(const SimdInt512& other) const {
  __mmask16 pcmp = _mm512_cmpeq_epi32_mask(ToVector(*this), ToVector(other));
  return pcmp == 0xffff;
}

template <>
SimdInt512 SimdInt512::operator&(const SimdInt512& other) const {
  return FromVector(_mm512_and_si512(ToVector(*this), ToVector(other)));
}

template <>
SimdInt512 SimdInt512::operator|(const SimdInt512& other) const {
  return FromVector(_mm512_or_si512(ToVector(*this), ToVector(other)));
}

template <>
SimdInt512 SimdInt512::operator^(const SimdInt512& other) const {
  return FromVector(_mm512_xor_si512(ToVector(*this), ToVector(other)));
}

template <>
SimdInt512 SimdInt512::operator>>(uint32_t count) const {
  if (UNLIKELY(count == 0)) return *this;
  if (UNLIKELY(count >= 512)) return SimdInt512();
  // TODO(chokobole): Optimize this.
  return SimdInt512(value_ >> count);
}

template <>
SimdInt512 SimdInt512::operator<<(uint32_t count) const {
  if (UNLIKELY(count == 0)) return *this;
  if (UNLIKELY(count >= 512)) return SimdInt512();
  // TODO(chokobole): Optimize this.
  return SimdInt512(value_ << count);
}

}  // namespace tachyon::math
