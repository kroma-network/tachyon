// Copyright 2024 Ulvetanna Inc.
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.ulvetanna file.

#include <immintrin.h>

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/compiler_specific.h"
#include "tachyon/math/base/simd_int.h"

namespace tachyon::math {

namespace {

__m256i ToVector(const SimdInt256& value) {
  return base::bit_cast<__m256i>(value);
}

SimdInt256 FromVector(__m256i vector) {
  SimdInt256 ret;
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(&ret), vector);
  return ret;
}

}  // namespace

// static
template <>
SimdInt256 SimdInt256::Broadcast(uint8_t value) {
  return FromVector(_mm256_set1_epi8(value));
}

// static
template <>
SimdInt256 SimdInt256::Broadcast(uint16_t value) {
  return FromVector(_mm256_set1_epi16(value));
}

// static
template <>
SimdInt256 SimdInt256::Broadcast(uint32_t value) {
  return FromVector(_mm256_set1_epi32(value));
}

// static
template <>
SimdInt256 SimdInt256::Broadcast(uint64_t value) {
  return FromVector(_mm256_set1_epi64x(value));
}

template <>
bool SimdInt256::operator==(const SimdInt256& other) const {
  __m256i pcmp = _mm256_cmpeq_epi32(ToVector(*this), ToVector(other));
  unsigned int bitmask =
      base::bit_cast<unsigned int>(_mm256_movemask_epi8(pcmp));
  return bitmask == 0xffffffff;
}

template <>
SimdInt256 SimdInt256::operator&(const SimdInt256& other) const {
  return FromVector(_mm256_and_si256(ToVector(*this), ToVector(other)));
}

template <>
SimdInt256 SimdInt256::operator|(const SimdInt256& other) const {
  return FromVector(_mm256_or_si256(ToVector(*this), ToVector(other)));
}

template <>
SimdInt256 SimdInt256::operator^(const SimdInt256& other) const {
  return FromVector(_mm256_xor_si256(ToVector(*this), ToVector(other)));
}

template <>
SimdInt256 SimdInt256::operator>>(uint32_t count) const {
  if (UNLIKELY(count == 0)) return *this;
  if (UNLIKELY(count >= 256)) return SimdInt256();
  // TODO(chokobole): Optimize this.
  return SimdInt256(value_ >> count);
}

template <>
SimdInt256 SimdInt256::operator<<(uint32_t count) const {
  if (UNLIKELY(count == 0)) return *this;
  if (UNLIKELY(count >= 256)) return SimdInt256();
  // TODO(chokobole): Optimize this.
  return SimdInt256(value_ << count);
}

}  // namespace tachyon::math
