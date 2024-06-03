// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS12_BLS12_CURVE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS12_BLS12_CURVE_H_

#include <functional>
#include <vector>

#include "tachyon/base/optional.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/math/elliptic_curves/bls12/g2_prepared.h"
#include "tachyon/math/elliptic_curves/pairing/pairing_friendly_curve.h"

namespace tachyon::math {

template <typename Config>
class BLS12Curve : public PairingFriendlyCurve<Config> {
 public:
  using Base = PairingFriendlyCurve<Config>;
  using Fp12 = typename Config::Fp12;
  using G2Prepared = bls12::G2Prepared<Config>;

  // TODO(chokobole): Leave a comment to help understand readers.
  template <typename G1AffinePointContainer, typename G2PreparedContainer>
  static Fp12 MultiMillerLoop(const G1AffinePointContainer& a,
                              const G2PreparedContainer& b) {
    using Pair = typename Base::Pair;

    std::vector<Pair> pairs = Base::CreatePairs(a, b);

    auto callback = [](absl::Span<const Pair> pairs) {
      Fp12 f = Fp12::One();
      auto it = BitIteratorBE<BigInt<Config::kXLimbNums>>::begin(
          &Config::kX,
          /*skip_leading_zeros=*/true);
      ++it;
      auto end = BitIteratorBE<BigInt<Config::kXLimbNums>>::end(&Config::kX);

      while (it != end) {
        f.SquareInPlace();

        for (const Pair& pair : pairs) {
          Base::Ell(f, pair.NextEllCoeff(), pair.g1());
        }

        if ((*it)) {
          for (const Pair& pair : pairs) {
            Base::Ell(f, pair.NextEllCoeff(), pair.g1());
          }
        }
        ++it;
      }
      return f;
    };

    std::vector<Fp12> results =
        base::ParallelizeMapByChunkSize(pairs, 4, callback);
    Fp12 f = std::accumulate(results.begin(), results.end(), Fp12::One(),
                             std::multiplies<>());

    if constexpr (Config::kXIsNegative) {
      CHECK(f.CyclotomicInverseInPlace());
    }
    return f;
  }

  static Fp12 FinalExponentiation(const Fp12& f) {
    // Computing the final exponentiation following
    // https://eprint.iacr.org/2020/875
    // Adapted from the implementation in
    // https://github.com/ConsenSys/gurvy/pull/29

    // f1 = f.CyclotomicInverseInPlace() = f^(q⁶)
    Fp12 f1 = f;
    CHECK(f1.CyclotomicInverseInPlace());

    // f2 = f⁻¹
    Fp12 f2 = unwrap<Fp12>(f.Inverse());

    // r = f^(q⁶ - 1)
    Fp12 r = f1 * f2;

    // f2 = f^(q⁶ - 1)
    f2 = r;
    // r = f^((q⁶ - 1)(q²))
    r.FrobeniusMapInPlace(2);

    // r = f^((q⁶ - 1)(q²)) * f^(q⁶ - 1)
    // r = f^((q⁶ - 1)(q² + 1))
    r *= f2;

    // Hard part of the final exponentiation:
    // y0 = r²
    Fp12 y0 = r.CyclotomicSquare();
    // y1 = (r)ˣ
    Fp12 y1 = Base::PowByX(r);
    // y2 = (r)⁻¹
    Fp12 y2 = unwrap<Fp12>(r.CyclotomicInverse());
    // y1 = y1 * y2 = r^(x - 1)
    y1 *= y2;
    // y2 = (y1)ˣ = r^(x² - x)
    y2 = Base::PowByX(y1);
    // y1 = (y1)⁻¹ = r^(-x + 1)
    CHECK(y1.CyclotomicInverseInPlace());
    // y1 = y1 * y2 = r^(x² - 2x + 1)
    y1 *= y2;
    // y2 = (y1)ˣ = r^(x³ - 2x² + x)
    y2 = Base::PowByX(y1);
    // y1 = (y1)^q = r^(q * (x² - 2x + 1))
    y1.FrobeniusMapInPlace(1);
    // y1 = y1 * y2 = r^(q * (x² - 2x  + 1) +
    //                   1 * (x³ - 2x² + x))
    y1 *= y2;
    // r = r * y0 = r³
    r *= y0;
    // y0 = (y1)ˣ = r^(q * (x³ - 2x² + x) +
    //                 1 * (x⁴ - 2x³ + x²))
    y0 = Base::PowByX(y1);
    // y2 = (y0)ˣ = r^(q * (x⁴ - 2x³ + x²) +
    //                 1 * (x⁵ - 2x⁴ + x³))
    y2 = Base::PowByX(y0);
    // y0 = (y1)^(q²) = r^(q³ * (x² - 2x  + 1)) +
    //                     q² * (x³ - 2x² + x))
    y0 = y1;
    y0.FrobeniusMapInPlace(2);
    // y1 = (y1)⁻¹ = r^(q * (-x² + 2x  - 1)) +
    //                  1 * (-x³ + 2x² - x))
    CHECK(y1.CyclotomicInverseInPlace());
    // y1 = y1 * y2 = r^(q * (x⁴ - 2x³ + 2x  - 1)) +
    //                   1 * (x⁵ - 2x⁴ + 2x² - x))
    y1 *= y2;
    // y1 = y1 * y0 = r^(q³ * (x² - 2x  +  1)) +
    //                   q² * (x³ - 2x² +  x))
    //                   q  * (x⁴ - 2x³ + 2x  - 1)) +
    //                   1  * (x⁵ - 2x⁴ + 2x² - x))
    y1 *= y0;
    // r = r * y1 = r^(q³ * (x² - 2x  +  1)) +
    //                 q² * (x³ - 2x² +  x))
    //                 q  * (x⁴ - 2x³ + 2x  - 1)) +
    //                 1  * (x⁵ - 2x⁴ + 2x² - x + 1))
    r *= y1;
    return r;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS12_BLS12_CURVE_H_
