// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

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

ALWAYS_INLINE int32x4_t MulByInv(int32x4_t val, int32x4_t inv) {
  // We want this to compile to:
  //      mul      r.4s, val.4s, inv.4s
  // throughput: .25 cyc/vec (16 els/cyc)
  // latency: 3 cyc

  return vmulq_s32(val, inv);
}

ALWAYS_INLINE int32x4_t GetCHi(int32x4_t lhs, int32x4_t rhs) {
  // We want this to compile to:
  //      sqdmulh  c_hi.4s, lhs.4s, rhs.4s
  // throughput: .25 cyc/vec (16 els/cyc)
  // latency: 3 cyc

  // Get bits 31, ..., 62 of C. Note that |sqdmulh| saturates when the product
  // doesn't fit in a 63 bit integer, but this cannot happen here due to our
  // bounds on |lhs| and |rhs|.
  return vqdmulhq_s32(lhs, rhs);
}

ALWAYS_INLINE int32x4_t GetQPHi(int32x4_t lhs, int32x4_t inv_rhs, int32x4_t p) {
  // We want this to compile to:
  //      mul      q.4s, lhs.4s, inv_rhs.4s
  //      sqdmulh  qp_hi.4s, q.4s, p.4s
  // throughput: .5 cyc/vec (8 els/cyc)
  // latency: 6 cyc

  int32x4_t q = vmulq_s32(lhs, inv_rhs);

  // Gets bits 31, ..., 62 of q * p. Again, saturation is not an issue because
  // |p| is not -2³¹.
  return vqdmulhq_s32(q, vreinterpretq_s32_u32(p));
}

ALWAYS_INLINE int32x4_t GetD(int32x4_t c_hi, int32x4_t qp_hi) {
  // We want this to compile to:
  //      shsub    r.4s, c_hi.4s, qp_hi.4s
  // throughput: .25 cyc/vec (16 els/cyc)
  // latency: 2 cyc

  // Form D. Note that |c_hi| is C >> 31 and |qp_hi| is (Q * P) >> 31, whereas
  // we want (C - Q * P) >> 32, so we need to subtract and divide by 2. Luckily
  // NEON has an instruction for that! The lowest bit of |c_hi| and |qp_hi| is
  // the same, so the division is exact.
  return vhsubq_s32(c_hi, qp_hi);
}

ALWAYS_INLINE uint32x4_t GetReducedD(int32x4_t c_hi, int32x4_t qp_hi,
                                     int32x4_t p) {
  // We want this to compile to:
  //      shsub    r.4s, c_hi.4s, qp_hi.4s
  //      cmgt     underflow.4s, qp_hi.4s, c_hi.4s
  //      mls      r.4s, underflow.4s, p.4s
  // throughput: .75 cyc/vec (5.33 els/cyc)
  // latency: 5 cyc

  uint32x4_t d = vreinterpretq_u32_s32(GetD(c_hi, qp_hi));

  // Finally, we reduce D to canonical form. D is negative iff |c_hi| > |qp_hi|,
  // so if that's the case then we add P. Note that if |c_hi| > |qp_hi| then
  // |underflow| is -1, so we must subtract |underflow * p|.
  uint32x4_t underflow = vcltq_s32(c_hi, qp_hi);
  // TODO(chokobole): add |ConfuseCompiler()|.
  // See
  // https://github.com/Plonky3/Plonky3/blob/6034010/baby-bear/src/aarch64_neon/packing.rs#L279.
  return vmlsq_u32(d, underflow, p);
}

ALWAYS_INLINE uint32x4_t MontMulMod32(uint32x4_t lhs, uint32x4_t rhs,
                                      uint32x4_t p, uint32x4_t inv) {
  // We want this to compile to:
  //      sqdmulh  c_hi.4s, lhs.4s, rhs.4s
  //      mul      inv_rhs.4s, rhs.4s, Inv.4s
  //      mul      q.4s, lhs.4s, inv_rhs.4s
  //      sqdmulh  qp_hi.4s, q.4s, p.4s
  //      shsub    r.4s, c_hi.4s, qp_hi.4s
  //      cmgt     underflow.4s, qp_hi.4s, c_hi.4s
  //      mls      r.4s, underflow.4s, p.4s
  // throughput: 1.75 cyc/vec (2.29 els/cyc)
  // latency: (lhs->) 11 cyc, (rhs->) 14 cyc

  // See comments "MONTGOMERY MULTIPLICATION" in
  // "tachyon/math/finite_fields/packed_prime_field_avx2.h".
  // No-op. The inputs are non-negative so we're free to interpret them as
  // signed numbers.
  int32x4_t s_lhs = vreinterpretq_s32_u32(lhs);
  int32x4_t s_rhs = vreinterpretq_s32_u32(rhs);

  uint32x4_t inv_rhs = MulByInv(s_rhs, inv);
  uint32x4_t c_hi = GetCHi(s_lhs, s_rhs);
  uint32x4_t qp_hi = GetQPHi(s_lhs, inv_rhs, p);
  return GetReducedD(c_hi, qp_hi, p);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_NEON_H_
