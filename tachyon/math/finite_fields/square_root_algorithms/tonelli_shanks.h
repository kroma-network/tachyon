// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_SQUARE_ROOT_ALGORITHMS_TONELLI_SHANKS_H_
#define TACHYON_MATH_FINITE_FIELDS_SQUARE_ROOT_ALGORITHMS_TONELLI_SHANKS_H_

#include <utility>

namespace tachyon::math {

template <typename F>
constexpr bool ComputeTonelliShanksSquareRoot(
    const F& a, const F& quadratic_non_residue_to_trace, F* ret) {
  // Fins x such that x² = a.
  // Here. modulus M is 2ˢ * T + 1. (where s is two adicity and T is trace).
  // https://eprint.iacr.org/2012/685.pdf (page 12, algorithm 5)
  if (a.IsZero()) {
    *ret = F::Zero();
    return true;
  }

  // Note that if a has a square root(in other words, a is a quadratic residue),
  // a^((M - 1) / 2) = 1 by Euler's criterion,
  // See https://en.wikipedia.org/wiki/Euler%27s_criterion
  // a^((M - 1) / 2) = a^(2ˢ⁻¹ * T) = (aᵀ)^(2ˢ⁻¹) = 1
  // aᵀ is 1 or 2ˢ⁻¹-th root of unity.

  // If we try
  // aᵀ * a = (a^((T + 1) / 2))^2
  // and if aᵀ is 1, then we can say the square root of a is a^((T + 1) / 2).
  F w = a.Pow(F::Config::kTraceMinusOneDivTwo);
  // x = aw = a^((T + 1) / 2)
  F x = w * a;
  // b = xw = aᵀ
  F b = x * w;

  if (!b.IsOne()) {
    // Otherwise, let's find a pair of x and b such that it satisfies
    // 1) x² = a * b
    // 2) b is 2ᵏ⁻¹-th root of unity. (1st iteration: b = aᵀ and k = s)
    // until b is 1.

    // z = cᵀ (where c is a non quadratic residue).
    // z^(2ᵛ⁻¹) = c^(2ᵛ⁻¹ * T) = c^((2ᵛ * T) / 2) = c^((M - 1) / 2) = -1
    // (since v = s)
    F z = quadratic_non_residue_to_trace;
    // v = s
    size_t v = size_t{F::Config::kTwoAdicity};
    do {
      size_t k = 0;

      // Find least integer k >= 0 such that b^(2ᵏ) = 1.
      F b2k = b;
      while (!b2k.IsOne()) {
        // invariant: b2k = b^(2ᵏ) after entering this loop
        b2k.SquareInPlace();
        ++k;
      }

      if (k == size_t{F::Config::kTwoAdicity}) {
        // We are in the case where a^(2ˢ * T) = xᴹ⁻¹ = 1,
        // which means that no square root exists.
        return false;
      }

      size_t j = v - k;
      // w = z^(2ᵛ⁻ᵏ⁻¹)
      w = z;
      for (size_t i = 1; i < j; ++i) {
        w.SquareInPlace();
      }

      // clang-format off
      // We have to find w and we replace x and b with xw and b * w²
      // This holds 1) because:
      // (x')² = (xw)² = x² * w² = a * b * w² = a * b'
      // This also holds 2) because:
      // (b')^(2ᵏ⁻¹) = b^(2ᵏ⁻¹) * (w²)^(2ᵏ⁻¹)
      //
      //   a) b^(2ᵏ⁻¹) = -1 because:
      //      b^(2ᵏ) = 1
      //      (b^(2ᵏ⁻¹) - 1) * (b^(2ᵏ⁻¹) + 1) = 0
      //      b^(2ᵏ⁻¹) = -1 (b^(2ᵏ⁻¹) can't be 1(since b is 2ᵏth root of unity)) <- Halving lemma
      //
      //   b) w²^(2ᵏ⁻¹) = -1 because:
      //      ((z^(2ᵛ⁻ᵏ⁻¹))^2)^(2ᵏ⁻¹) = (z^(2ᵛ⁻ᵏ))^(2ᵏ⁻¹) = z^(2ᵛ⁻¹) = -1 (See above why)
      //
      // Therefore, b' is 2ᵏ⁻¹ th root of unity.
      // clang-format on

      // z = w²
      z = w.Square();
      // b = bz
      b *= z;
      // x = xw
      x *= w;
      // v = k
      v = k;
    } while (!b.IsOne());
  }

  if (x.Square() == a) {
    *ret = std::move(x);
    return true;
  }
  return false;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_SQUARE_ROOT_ALGORITHMS_TONELLI_SHANKS_H_
