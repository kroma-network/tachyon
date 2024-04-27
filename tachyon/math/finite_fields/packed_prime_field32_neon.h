#ifndef TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_NEON_H_
#define TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_NEON_H_

#include <arm_neon.h>

#include "tachyon/base/compiler_specific.h"

namespace tachyon::math {

// Given a |val| in {0, ..., 2p}, return a |res| in {0, ..., p} such
// that |res = val (mod p)|.
ALWAYS_INLINE uint32x4_t ReduceSum32(uint32x4_t val, uint32x4_t p) {
  // NOTE(chokobole): This assumes 2p < 2³², where p is the modulus.
  // Let u := (val - p) mod 2³² and r := min(val, u).
  // 0 ≤ val ≤ 2p
  //
  // 1) 0 ≤ val ≤ p - 1
  //    p < 2³² - p ≤ u ≤ 2³² - 1
  //    r = val
  //
  // 2) p ≤ val ≤ 2p
  //    0 ≤ u ≤ p
  //    r = u
  //
  // In both cases, r is in {0, ..., p}.
  uint32x4_t u = vsubq_u32(val, p);
  return vminq_u32(val, u);
}

ALWAYS_INLINE uint32x4_t AddMod32(uint32x4_t lhs, uint32x4_t rhs,
                                  uint32x4_t p) {
  // NOTE(chokobole): This assumes 2p < 2³², where p is the modulus.
  // We want this to compile to:
  //      add   t.4s, lhs.4s, rhs.4s
  //      sub   u.4s, t.4s, p.4s
  //      umin  r.4s, t.4s, u.4s
  // throughput: .75 cyc/vec (5.33 els/cyc)
  // latency: 6 cyc

  // Let t := lhs + rhs
  //     u := (t - p) mod 2³²
  //     r := min(t, u)
  //     m := { p     if this is montgomery form, which is 0
  //          { p - 1 otherwise
  //
  // 0 ≤ lhs, rhs ≤ m
  //
  // 1) (lhs = p && rhs = 0) || (lhs = 0 && rhs = p)
  //    t = p
  //    u = 0
  //    r = 0
  //
  // 2) lhs = p && rhs = p
  //    t = 2p
  //    u = p
  //    r = p, which is 0.
  //
  // 3) (lhs = p && 1 ≤ rhs ≤ p - 1) || (1 ≤ lhs ≤ p - 1 && rhs = p)
  //    p + 1 ≤ t ≤ 2p - 1, go to 5)
  //
  // 4) 0 ≤ t ≤ p - 1
  //    p < 2³² - p ≤ u ≤ 2³² - 1
  //    r = t
  //
  // 5) p + 1 ≤ t ≤ 2p - 1
  //    1 ≤ u ≤ p - 1 ≤ p
  //    r = u
  //
  // In all cases, r is in {0, ..., m}.
  uint32x4_t t = vaddq_u32(lhs, rhs);
  return ReduceSum32(t, p);
}

ALWAYS_INLINE uint32x4_t SubMod32(uint32x4_t lhs, uint32x4_t rhs,
                                  uint32x4_t p) {
  // NOTE(chokobole): This assumes 2p < 2³², where p is the modulus.
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
  //     m := { p     if this is montgomery form, which is 0
  //          { p - 1 otherwise
  //
  // 0 ≤ lhs, rhs ≤ m
  //
  // 1) lhs = p && rhs = 0
  //    t = p
  //    diff = p
  //    underflow = 0
  //    r = diff = p, which is 0.
  //
  // 2) lhs = 0 && rhs = p
  //    t = -p
  //    diff = 2³² - p
  //    underflow = 2³² - 1
  //    r = 2³² - p - (2³² - 1)p = 0
  //
  // 3) lhs = p && rhs = p
  //    t = 0
  //    diff = 0
  //    underflow = 0
  //    r = diff = 0
  //
  // 4) lhs = p && 1 ≤ rhs ≤ p - 1
  //    1 ≤ t ≤ p - 1, go to 6)
  //
  // 5) 1 ≤ lhs ≤ p - 1 && rhs = p
  //    -p + 1 ≤ t ≤ -1, go to 7)
  //
  // 6) 1 ≤ t ≤ p - 1
  //    1 ≤ diff ≤ p - 1
  //    underflow = 0
  //    r = t
  //
  // 7) -p + 1 ≤ t ≤ -1
  //    2³² - p + 1 ≤ diff ≤ 2³² - 1
  //    underflow = 2³² - 1
  //    2³² - p + 1 - (2³² - 1)p ≤ r ≤ 2³² - 1 - (2³² - 1)p
  //    1 ≤ r ≤ p - 1
  //
  // In all cases, r is in {0, ..., m}.
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
  //     m := { p     if this is montgomery form, which is 0
  //          { p - 1 otherwise
  //
  // 0 ≤ val ≤ m
  //
  // 1) val = 0
  //    is_zero = 2³² - 1
  //    r = 0
  //
  // 2) val = p
  //    is_zero = 0
  //    t = 0
  //    r = 0
  //
  // 3) 1 ≤ val ≤ p - 1
  //    is_zero = 0
  //    2³² - p + 1 ≤ -val ≤ 2³² - 1
  //    1 ≤ t ≤ p - 1
  //    r = t
  //
  // In all cases, r is in {0, ..., m}.
  uint32x4_t t = vsubq_u32(p, val);
  uint32x4_t is_zero = vceqzq_u32(val);
  return vbicq_u32(t, is_zero);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_NEON_H_
