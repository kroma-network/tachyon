// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_neon.h"

#include <arm_neon.h>

#include "tachyon/math/finite_fields/packed_prime_field32_neon.h"

namespace tachyon::math {

namespace {

uint32x4_t kP;
uint32x4_t kZero;
uint32x4_t kOne;

uint32x4_t ToVector(const PackedMersenne31Neon& packed) {
  return vld1q_u32(reinterpret_cast<const uint32_t*>(packed.values().data()));
}

PackedMersenne31Neon FromVector(uint32x4_t vector) {
  PackedMersenne31Neon ret;
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

// Multiply two 31-bit numbers to obtain a 62-bit immediate result, and return
// the high 31 bits of that result. Results are arbitrary if the inputs do not
// fit in 31 bits.
uint32x4_t mul_31x31_to_hi_31(uint32x4_t lhs, uint32x4_t rhs) {
  return vreinterpretq_u32_s32(
      vqdmulhq_s32(vreinterpretq_s32_u32(lhs), vreinterpretq_s32_u32(rhs)));
}

uint32x4_t Mul(uint32x4_t lhs, uint32x4_t rhs) {
  // We want this to compile to:
  //      sqdmulh  prod_hi31.4s, lhs.4s, rhs.4s
  //      mul      t.4s, lhs.4s, rhs.4s
  //      mla      t.4s, prod_hi31.4s, P.4s
  //      sub      u.4s, t.4s, P.4s
  //      umin     r.4s, t.4s, u.4s
  // throughput: 1.25 cyc/vec (3.2 els/cyc)
  // latency: 10 cyc

  // Let prod := lhs * rhs
  //     prod_hi31 := (prod >> 2³¹) (mod 2³¹)
  //               = a₃₁ + 2 * a₃₂ + 2² * a₃₃ + ... + 2³⁰ * a₆₂
  //     prod_lo31 := prod (mod 2³¹) = a₀ + 2 * a₁ + 2² * a₂ + ... + 2³⁰ * a₃₀
  //     prod_lo32 := prod (mod 2³²) = a₀ + 2 * a₁ + 2² * a₂ + ... + 2³¹ * a₃₁
  //     t := prod_hi31 + prod_lo31
  //
  //     prod = 2³¹ * prod_hi31 + prod_lo31 (mod P)
  //          = (2³¹ - 1) * prod_hi31 + prod_hi31 + prod_lo31 (mod P)
  //          = P * prod_hi31 + prod_hi31 + prod_lo31 (mod P)
  //          = prod_hi31 + prod_lo31 (mod P)
  //          = t (mod P)
  //
  //     prod_lo32 = prod_lo31 + 2³¹ (prod_hi31 mod 2)
  //               = prod_lo31 + 2³¹ prod_hi31 (mod 2³²)
  //
  //     2³¹ * prod_hi31 = 2³¹ (a₃₁ + 2 * a₃₂ + ... + 2³⁰ * a₆₂) (mod 2³²)
  //                     = 2³¹ * a₃₁ + 2³²(a₃₂ + ... + 2²⁹ * a₆₂) (mod 2³²)
  //                     = 2³¹ * a₃₁ (mod 2³²)
  //
  //     t = prod_lo31 + prod_hi31 (mod 2³²)
  //       = prod_lo32 - 2³¹ * a₃₁ + prod_hi31 (mod 2³²)
  //       = prod_lo32 - 2³¹ * prod_hi31 + prod_hi31 (mod 2³²)
  //       = prod_lo32 - (2³¹ - 1) prod_hi31 (mod 2³²)
  //       = prod_lo32 - P * prod_hi31 (mod 2³²)
  //
  // 0 ≤ t ≤ 2³² - 1 = 2(P + 1) - 1 = 2P - 1
  // So, ReduceSum32(t) is in range {0, ..., P - 1}.
  uint32x4_t prod_hi31 = mul_31x31_to_hi_31(lhs, rhs);
  uint32x4_t prod_lo32 = vmulq_u32(lhs, rhs);
  uint32x4_t t = vmlsq_u32(prod_lo32, prod_hi31, kP);
  return ReduceSum32(t, kP);
}

}  // namespace

PackedMersenne31Neon::PackedMersenne31Neon(uint32_t value) {
  uint32x4_t vector = vdupq_n_u32(value);
  vst1q_u32(reinterpret_cast<uint32_t*>(values_.data()), vector);
}

// static
void PackedMersenne31Neon::Init() {
  kP = vdupq_n_u32(Mersenne31::Config::kModulus);
  kZero = vdupq_n_u32(0);
  kOne = vdupq_n_u32(1);
}

// static
PackedMersenne31Neon PackedMersenne31Neon::Zero() { return FromVector(kZero); }

// static
PackedMersenne31Neon PackedMersenne31Neon::One() { return FromVector(kOne); }

// static
PackedMersenne31Neon PackedMersenne31Neon::Broadcast(const PrimeField& value) {
  return FromVector(vdupq_n_u32(value.value()));
}

PackedMersenne31Neon PackedMersenne31Neon::Add(
    const PackedMersenne31Neon& other) const {
  return FromVector(math::Add(ToVector(*this), ToVector(other)));
}

PackedMersenne31Neon PackedMersenne31Neon::Sub(
    const PackedMersenne31Neon& other) const {
  return FromVector(math::Sub(ToVector(*this), ToVector(other)));
}

PackedMersenne31Neon PackedMersenne31Neon::Negate() const {
  return FromVector(math::Negate(ToVector(*this)));
}

PackedMersenne31Neon PackedMersenne31Neon::Mul(
    const PackedMersenne31Neon& other) const {
  return FromVector(math::Mul(ToVector(*this), ToVector(other)));
}

}  // namespace tachyon::math
