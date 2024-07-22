// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/math/finite_fields/koala_bear/packed_koala_bear_neon.h"

#include <arm_neon.h>

#include "tachyon/math/finite_fields/packed_prime_field32_neon.h"

namespace tachyon::math {

namespace {

uint32x4_t kP;
uint32x4_t kInv;
uint32x4_t kZero;
uint32x4_t kOne;
uint32x4_t kMinusOne;

uint32x4_t ToVector(const PackedKoalaBearNeon& packed) {
  return vld1q_u32(reinterpret_cast<const uint32_t*>(packed.values().data()));
}

PackedKoalaBearNeon FromVector(uint32x4_t vector) {
  PackedKoalaBearNeon ret;
  vst1q_u32(reinterpret_cast<uint32_t*>(ret.values().data()), vector);
  return ret;
}

uint32x4_t Add(uint32x4_t lhs, uint32x4_t rhs) {
  return AddMod32(lhs, rhs, kP);
}

uint32x4_t Sub(uint32x4_t lhs, uint32x4_t rhs) {
  return SubMod32(lhs, rhs, kP);
}

uint32x4_t Negate(uint32x4_t val) { return NegateMod32(val, kP); }

uint32x4_t Mul(uint32x4_t lhs, uint32x4_t rhs) {
  return MontMulMod32(lhs, rhs, kP, kInv);
}

}  // namespace

PackedKoalaBearNeon::PackedKoalaBearNeon(uint32_t value) {
  uint32x4_t vector = vdupq_n_u32(KoalaBear::Config::ToMontgomery(value));
  vst1q_u32(reinterpret_cast<uint32_t*>(values_.data()), vector);
}

// static
void PackedKoalaBearNeon::Init() {
  KoalaBear::Init();
  kP = vdupq_n_u32(KoalaBear::Config::kModulus);
  kInv = vdupq_n_u32(KoalaBear::Config::kInverse32);
  kZero = vdupq_n_u32(0);
  kOne = vdupq_n_u32(KoalaBear::Config::kOne);
  kMinusOne = vdupq_n_u32(KoalaBear::Config::kMinusOne);
}

// static
PackedKoalaBearNeon PackedKoalaBearNeon::Zero() { return FromVector(kZero); }

// static
PackedKoalaBearNeon PackedKoalaBearNeon::One() { return FromVector(kOne); }

// static
PackedKoalaBearNeon PackedKoalaBearNeon::MinusOne() {
  return FromVector(kMinusOne);
}

// static
PackedKoalaBearNeon PackedKoalaBearNeon::Broadcast(const PrimeField& value) {
  return FromVector(vdupq_n_u32(value.value()));
}

PackedKoalaBearNeon PackedKoalaBearNeon::Add(
    const PackedKoalaBearNeon& other) const {
  return FromVector(math::Add(ToVector(*this), ToVector(other)));
}

PackedKoalaBearNeon PackedKoalaBearNeon::Sub(
    const PackedKoalaBearNeon& other) const {
  return FromVector(math::Sub(ToVector(*this), ToVector(other)));
}

PackedKoalaBearNeon PackedKoalaBearNeon::Negate() const {
  return FromVector(math::Negate(ToVector(*this)));
}

PackedKoalaBearNeon PackedKoalaBearNeon::Mul(
    const PackedKoalaBearNeon& other) const {
  return FromVector(math::Mul(ToVector(*this), ToVector(other)));
}

}  // namespace tachyon::math
