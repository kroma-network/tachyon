// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN_CURVE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN_CURVE_H_

#include <functional>
#include <vector>

#include "tachyon/base/parallelize.h"
#include "tachyon/math/elliptic_curves/bn/g2_prepared.h"
#include "tachyon/math/elliptic_curves/pairing/pairing_friendly_curve.h"

namespace tachyon::math {

template <typename Config>
class BNCurve : public PairingFriendlyCurve<Config> {
 public:
  using Base = PairingFriendlyCurve<Config>;
  using Fp12Ty = typename Config::Fp12Ty;
  using G2Prepared = bn::G2Prepared<Config>;

  // TODO(chokobole): Leave a comment to help understand readers.
  template <typename G1AffinePointContainer, typename G2PreparedContainer>
  static Fp12Ty MultiMillerLoop(const G1AffinePointContainer& a,
                                const G2PreparedContainer& b) {
    using Pair = typename Base::Pair;

    std::vector<Pair> pairs = Base::CreatePairs(a, b);

    auto callback = [](absl::Span<const Pair> pairs) {
      Fp12Ty f = Fp12Ty::One();
      for (size_t i = std::size(Config::kAteLoopCount) - 1; i >= 1; --i) {
        if (i != std::size(Config::kAteLoopCount) - 1) {
          f.SquareInPlace();
        }

        for (const Pair& pair : pairs) {
          Base::Ell(f, pair.NextEllCoeff(), pair.g1());
        }

        int8_t bit = Config::kAteLoopCount[i - 1];
        if (bit == 1 || bit == -1) {
          for (const Pair& pair : pairs) {
            Base::Ell(f, pair.NextEllCoeff(), pair.g1());
          }
        }
      }
      return f;
    };

    std::vector<Fp12Ty> results =
        base::ParallelizeMapByChunkSize(pairs, 4, callback);
    Fp12Ty f = std::accumulate(results.begin(), results.end(), Fp12Ty::One(),
                               std::multiplies<>());

    if constexpr (Config::kXIsNegative) {
      f.CyclotomicInverseInPlace();
    }

    for (const Pair& pair : pairs) {
      Base::Ell(f, pair.NextEllCoeff(), pair.g1());
    }

    for (const Pair& pair : pairs) {
      Base::Ell(f, pair.NextEllCoeff(), pair.g1());
    }

    return f;
  }

  static Fp12Ty FinalExponentiation(const Fp12Ty& f) {
    // clang-format off
    // f^((q¹² - 1) / r) = f^((q⁶ - 1) * ((q⁶ + 1) / φ₁₂(q)) * (φ₁₂(q) / r))
    //                   = f^((q⁶ - 1) * ((q⁶ + 1) / (q⁴ - q² + 1)) * ((q⁴ - q² + 1) / r))
    //                   = f^((q⁶ - 1) * (q² + 1) * ((q⁴ - q² + 1) / r))
    //                       <--- easy part --->    <--- hard part --->
    // clang-format on
    // Easy part: result = f^((q⁶ - 1) * (q² + 1)).
    // Follows, e.g., Beuchat et al page 9, by computing result as follows:
    //   f^((q⁶ - 1) * (q² + 1)) = (conj(f) * f⁻¹)^(q² + 1)

    // f1 = f.CyclotomicInverseInPlace() = f^(q⁶)
    Fp12Ty f1 = f;
    f1.CyclotomicInverseInPlace();

    // f2 = f⁻¹
    Fp12Ty f2 = f.Inverse();

    // r = f^(q⁶ - 1)
    Fp12Ty r = f1 * f2;

    // f2 = f^(q⁶ - 1)
    f2 = r;
    // r = f^((q⁶ - 1)(q²))
    r.FrobeniusMapInPlace(2);

    // r = f^((q⁶ - 1)(q²)) * f^(q⁶ - 1)
    // r = f^((q⁶ - 1)(q² + 1))
    r *= f2;

    // Hard part follows Laura Fuentes-Castaneda et al. "Faster hashing to G2"
    // by computing:
    //
    // result = f^(q³ * (12x³ +  6x² + 4x - 1) +
    //             q² * (12x³ +  6x² + 6x    ) +
    //             q  * (12x³ +  6x² + 4x    ) +
    //             1  * (12x³ + 12x² + 6x + 1))
    // which equals
    //
    // TODO(chokobole): Add easier comment after understanding what below means.
    // I just guess that 2x * (6x² + 3x + 1) equals to (q⁶ - 1)(q² + 1).
    // result = f^(2x * (6x² + 3x + 1) * ((q⁴ - q² + 1) / r))

    // y0 = (r)⁻ˣ
    Fp12Ty y0 = Base::PowByNegX(r);
    // y1 = (y0)² = r^(-2x)
    Fp12Ty& y1 = y0.CyclotomicSquareInPlace();
    // y2 = (y1)² = r^(-4x)
    Fp12Ty y2 = y1.CyclotomicSquare();
    // y3 = y2 * y1 = r^(-6x)
    Fp12Ty& y3 = y2 *= y1;
    // y4 = (y3)⁻ˣ = r^(6x²)
    Fp12Ty y4 = Base::PowByNegX(y3);
    // y5 = (y4)² = r^(12x²)
    Fp12Ty y5 = y4.CyclotomicSquare();
    // y6 = (y5)⁻ˣ = r^(-12x³)
    Fp12Ty y6 = Base::PowByNegX(y5);
    // y3 = (y3)⁻¹ = r^(6x)
    y3.CyclotomicInverseInPlace();
    // y6 = (y6)⁻¹ = r^(12x³)
    y6.CyclotomicInverseInPlace();
    // y7 = y6 * y4 = r^(12x³ + 6x²)
    Fp12Ty& y7 = y6 *= y4;
    // y8 = y7 * y3 = r^(12x³ + 6x² + 6x)
    Fp12Ty y8 = y7 * y3;
    // y9 = y8 * y1 = r^(12x³ + 6x² + 4x)
    Fp12Ty y9 = y8 * y1;
    // y10 = y8 * y4 = r^(12x³ + 12x² + 6x)
    Fp12Ty y10 = y8 * y4;
    // y11 = y10 * r = r^(12x³ + 12x² + 6x + 1)
    Fp12Ty& y11 = y10 *= r;
    // y12 = (y9)^q = r^(q * (12x³ + 6x² + 4x))
    Fp12Ty y12 = y9;
    y12.FrobeniusMapInPlace(1);
    // y13 = y12 * y11 = r^(q * (12x³ +  6x² + 4x) +
    //                      1 * (12x³ + 12x² + 6x + 1))
    Fp12Ty& y13 = y12 *= y11;
    // y8 = (y8)^q² = r^(q² * (12x³ + 6x² + 6x))
    y8.FrobeniusMapInPlace(2);
    // y14 = y8 * y13 = r^(q² * (12x³ +  6x² + 6x) +
    //                     q  * (12x³ +  6x² + 4x) +
    //                     1  * (12x³ + 12x² + 6x + 1))
    Fp12Ty& y14 = y8 *= y13;
    r.CyclotomicInverseInPlace();
    // y15 = r⁻¹ * y9 = r^(12x³ + 6x² + 4x - 1)
    Fp12Ty& y15 = r *= y9;
    // y15 = (y15)^q³ = r^(q³ * (12x³ + 6x² + 4x - 1))
    y15.FrobeniusMapInPlace(3);
    // y16 = y15 * y14 = r^(q³ * (12x³ +  6x² + 4x - 1) +
    //                      q² * (12x³ +  6x² + 6x) +
    //                      q  * (12x³ +  6x² + 4x) +
    //                      1  * (12x³ + 12x² + 6x + 1))
    Fp12Ty& y16 = y15 *= y14;
    return y16;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN_CURVE_H_
