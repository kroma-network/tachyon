// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS12_BLS12_CURVE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS12_BLS12_CURVE_H_

namespace tachyon::math {

template <typename BLS12CurveConfig>
class BLS12Curve {
 public:
  using Config = BLS12CurveConfig;
  using Fp12Ty = typename Config::Fp12Ty;

  static Fp12Ty PowByX(const Fp12Ty& f_in) {
    Fp12Ty f = f_in.CyclotomicPow(Config::kX);
    if constexpr (Config::kXIsNegative) {
      f.CyclotomicInverseInPlace();
    }
    return f;
  }

  static Fp12Ty FinalExponentiation(const Fp12Ty& f) {
    // Computing the final exponentiation following
    // https://eprint.iacr.org/2020/875
    // Adapted from the implementation in
    // https://github.com/ConsenSys/gurvy/pull/29

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

    // Hard part of the final exponentiation:
    // y0 = r²
    Fp12Ty y0 = r.CyclotomicSquare();
    // y1 = (r)ˣ
    Fp12Ty y1 = PowByX(r);
    // y2 = (r)⁻¹
    Fp12Ty y2 = r.CyclotomicInverse();
    // y1 = y1 * y2 = r^(x - 1)
    y1 *= y2;
    // y2 = (y1)ˣ = r^(x² - x)
    y2 = PowByX(y1);
    // y1 = (y1)⁻¹ = r^(-x + 1)
    y1.CyclotomicInverseInPlace();
    // y1 = y1 * y2 = r^(x² - 2x + 1)
    y1 *= y2;
    // y2 = (y1)ˣ = r^(x³ - 2x² + x)
    y2 = PowByX(y1);
    // y1 = (y1)^q = r^(q * (x² - 2x + 1))
    y1.FrobeniusMapInPlace(1);
    // y1 = y1 * y2 = r^(q * (x² - 2x  + 1) +
    //                   1 * (x³ - 2x² + x))
    y1 *= y2;
    // r = r * y0 = r³
    r *= y0;
    // y0 = (y1)ˣ = r^(q * (x³ - 2x² + x) +
    //                 1 * (x⁴ - 2x³ + x²))
    y0 = PowByX(y1);
    // y2 = (y0)ˣ = r^(q * (x⁴ - 2x³ + x²) +
    //                 1 * (x⁵ - 2x⁴ + x³))
    y2 = PowByX(y0);
    // y0 = (y1)^(q²) = r^(q³ * (x² - 2x  + 1)) +
    //                     q² * (x³ - 2x² + x))
    y0 = y1;
    y0.FrobeniusMapInPlace(2);
    // y1 = (y1)⁻¹ = r^(q * (-x² + 2x  - 1)) +
    //                  1 * (-x³ + 2x² - x))
    y1.CyclotomicInverseInPlace();
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
