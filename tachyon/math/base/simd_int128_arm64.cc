// Copyright 2024 Ulvetanna Inc.
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.ulvetanna file.

#include <arm_neon.h>

#include <algorithm>

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/base/simd_int.h"

namespace tachyon::math {

namespace {

uint8x16_t ToVector(const SimdInt128& value) {
  return vld1q_u8(reinterpret_cast<const uint8_t*>(&value));
}

SimdInt128 FromVector(uint8x16_t vector) {
  SimdInt128 ret;
  vst1q_u8(reinterpret_cast<uint8_t*>(&ret), vector);
  return ret;
}

SimdInt128 FromVector(uint16x8_t vector) {
  SimdInt128 ret;
  vst1q_u16(reinterpret_cast<uint16_t*>(&ret), vector);
  return ret;
}

SimdInt128 FromVector(uint32x4_t vector) {
  SimdInt128 ret;
  vst1q_u32(reinterpret_cast<uint32_t*>(&ret), vector);
  return ret;
}

SimdInt128 FromVector(uint64x2_t vector) {
  SimdInt128 ret;
  vst1q_u64(reinterpret_cast<uint64_t*>(&ret), vector);
  return ret;
}

}  // namespace

// static
template <>
SimdInt128 SimdInt128::Broadcast(uint8_t value) {
  return FromVector(vdupq_n_u8(value));
}

// static
template <>
SimdInt128 SimdInt128::Broadcast(uint16_t value) {
  return FromVector(vdupq_n_u16(value));
}

// static
template <>
SimdInt128 SimdInt128::Broadcast(uint32_t value) {
  return FromVector(vdupq_n_u32(value));
}

// static
template <>
SimdInt128 SimdInt128::Broadcast(uint64_t value) {
  return FromVector(vdupq_n_u64(value));
}

template <>
bool SimdInt128::operator==(const SimdInt128& other) const {
  return value_ == other.value_;
}

template <>
SimdInt128 SimdInt128::operator&(const SimdInt128& other) const {
  return FromVector(vandq_u8(ToVector(*this), ToVector(other)));
}

template <>
SimdInt128 SimdInt128::operator|(const SimdInt128& other) const {
  return FromVector(vorrq_u8(ToVector(*this), ToVector(other)));
}

template <>
SimdInt128 SimdInt128::operator^(const SimdInt128& other) const {
  return FromVector(veorq_u8(ToVector(*this), ToVector(other)));
}

template <>
SimdInt128 SimdInt128::operator>>(uint32_t count) const {
  if (UNLIKELY(count == 0)) return *this;
  if (UNLIKELY(count >= 128)) return SimdInt128();
  return SimdInt128(value_ >> count);
}

template <>
SimdInt128 SimdInt128::operator<<(uint32_t count) const {
  if (UNLIKELY(count == 0)) return *this;
  if (UNLIKELY(count >= 128)) return SimdInt128();
  return SimdInt128(value_ << count);
}

}  // namespace tachyon::math
