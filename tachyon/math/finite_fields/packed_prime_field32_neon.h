#ifndef TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_NEON_H_
#define TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_NEON_H_

#include <arm_neon.h>

#include "tachyon/base/compiler_specific.h"

namespace tachyon::math {

// Given a |val| in {0, ..., 2p - 1}, return a |res| in {0, ..., p - 1} such
// that |res = val (mod p)|.
ALWAYS_INLINE uint32x4_t ReduceSum32(uint32x4_t val, uint32x4_t p) {
  // Let u := (val - p) mod 2³² and r := min(t, u).
  // 0 ≤ val ≤ 2p - 1
  //
  // 1) if 0 ≤ val ≤ p - 1:
  //    2³² - p ≤ u ≤ 2³² - 1
  //    2(p + 1) - p ≤ u ≤ 2³² - 1
  //    p - 1 < p + 1 ≤ u ≤ 2³² - 1
  //    r = t
  //
  // 2) otherwise p ≤ val ≤ 2p - 1:
  //    0 ≤ u ≤ p - 1 < p
  //    r = u
  //
  // In both cases, r is in {0, ..., p - 1}.
  uint32x4_t u = vsubq_u32(val, p);
  return vminq_u32(val, u);
}

ALWAYS_INLINE uint32x4_t AddMod32(uint32x4_t lhs, uint32x4_t rhs,
                                  uint32x4_t p) {
  // NOTE(chokobole): This assumes that the 2p - 2 < 2³², where p is modulus.
  // We want this to compile to:
  //      add   t.4s, lhs.4s, rhs.4s
  //      sub   u.4s, t.4s, p.4s
  //      umin  r.4s, t.4s, u.4s
  // throughput: .75 cyc/vec (5.33 els/cyc)
  // latency: 6 cyc

  // Let t := lhs + rhs
  //     u := (t - p) mod 2³²
  //     r := min(t, u)
  //
  // 0 ≤ lhs, rhs ≤ p - 1
  // 0 ≤ t ≤ 2p - 2
  //
  // 1) if 0 ≤ t ≤ p - 1:
  //    2³² - p ≤ u ≤ 2³² - 1
  //    2(p + 1) - p ≤ u ≤ 2³² - 1
  //    p - 1 < p + 1 ≤ u ≤ 2³² - 1
  //    r = t
  //
  // 2) otherwise p ≤ t ≤ 2p - 2:
  //    0 ≤ u ≤ p - 2 < p
  //    r = u
  //
  // In both cases, r is in {0, ..., p - 1}.
  uint32x4_t t = vaddq_u32(lhs, rhs);
  return ReduceSum32(t, p);
}

ALWAYS_INLINE uint32x4_t SubMod32(uint32x4_t lhs, uint32x4_t rhs,
                                  uint32x4_t p) {
  // NOTE(chokobole): This assumes that the 2p - 2 < 2³², where p is modulus.
  // We want this to compile to:
  //      sub   r.4s, lhs.4s, rhs.4s
  //      cmhi  underflow.4s, rhs.4s, lhs.4s
  //      mls   r.4s, underflow.4s, p.4s
  // throughput: .75 cyc/vec (5.33 els/cyc)
  // latency: 5 cyc

  // Let t := lhs - rhs
  //     diff := t mod 2³²
  //     underflow := 0 if lhs ≥ rhs else 2³² - 1
  //     r := (diff - underflow * p) mod 2³²
  //
  // 0 ≤ lhs, rhs ≤ p - 1
  //
  // 1) lhs ≥ rhs -> underflow = 0:
  //    0 ≤ t ≤ p - 1
  //    0 ≤ r = diff ≤ p - 1
  //
  // 2) otherwise lhs < rhs -> underflow = 2³² - 1:
  //    -p + 1 ≤ t ≤ -1
  //    2³² + -p + 1 ≤ diff ≤ 2³² -1
  //    2³² -p + 1 - (2³² - 1)p ≤ r ≤ 2³² - 1 - (2³² - 1)p
  //    2³²(1 - p) + 1 ≤ r ≤ 2³²(1 - p) + p - 1
  //    1 ≤ r ≤ p - 1
  //
  // In both cases, r is in {0, ..., p - 1}.
  uint32x4_t diff = vsubq_u32(lhs, rhs);
  uint32x4_t underflow = vcltq_u32(lhs, rhs);
  return vmlsq_u32(diff, underflow, p);
}

ALWAYS_INLINE uint32x4_t NegateMod32(uint32x4_t val, uint32x4_t p) {
  // We want this to compile to:
  //      sub   t.4s, p.4s, val.4s
  //      cmeq  is_zero.4s, val.4s, #0
  //      bic   r.4s, t.4s, is_zero.4s
  // throughput: .75 cyc/vec (5.33 els/cyc)
  // latency: 4 cyc

  // This has the same throughput as |Sub(kZero, val)| but slightly lower
  // latency.
  //
  // Let t := p - val
  //     is_zero := 2³² - 1 if val = 0 else 0
  //     r := t & ~is_zero
  //
  // 0 ≤ val ≤ p - 1
  //
  // 1) val = 0 -> is_zero = 2³² - 1:
  //    r = 0
  //
  // 2) otherwise 1 ≤ val ≤ p - 1 -> is_zero = 0:
  //    r = t
  uint32x4_t t = vsubq_u32(p, val);
  uint32x4_t is_zero = vceqzq_u32(val);
  return vbicq_u32(t, is_zero);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_NEON_H_
