// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP12_H_
#define TACHYON_MATH_FINITE_FIELDS_FP12_H_

#include <type_traits>
#include <utility>

#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/finite_fields/quadratic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp12 final : public QuadraticExtensionField<Fp12<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using Fp6 = BaseField;
  using Fp2 = typename Fp6::BaseField;

  using CpuField = Fp12<Config>;
  // TODO(chokobole): Implement Fp12Gpu
  using GpuField = Fp12<Config>;

  using QuadraticExtensionField<Fp12<Config>>::QuadraticExtensionField;

  static_assert(Config::kDegreeOverBaseField == 2);
  static_assert(BaseField::ExtensionDegree() == 6);

  constexpr static uint32_t kDegreeOverBasePrimeField = 12;

  static void Init() {
    using BaseFieldConfig = typename BaseField::Config;
    // x⁶ = q = |BaseFieldConfig::kNonResidue|

    Config::Init();

    // αᴾ = (α₀ + α₁x)ᴾ
    //    = α₀ᴾ + α₁ᴾxᴾ
    //    = α₀ᴾ + α₁ᴾxᴾ⁻¹x
    //    = α₀ᴾ + α₁ᴾ(x⁶)^((P - 1) / 6) * x
    //    = α₀ᴾ + α₁ᴾωx, where ω is a sextic root of unity.

    using UnpackedBasePrimeField = MaybeUnpack<BasePrimeField>;

    constexpr size_t N = UnpackedBasePrimeField::kLimbNums;
    // m₁ = P
    mpz_class m1;
    if constexpr (UnpackedBasePrimeField::Config::kModulusBits <= 32) {
      m1 = mpz_class(UnpackedBasePrimeField::Config::kModulus);
    } else {
      gmp::WriteLimbs(UnpackedBasePrimeField::Config::kModulus.limbs, N, &m1);
    }

#define SET_M(d, d_prev) mpz_class m##d = m##d_prev * m1

    // m₂ = m₁ * P = P²
    SET_M(2, 1);
    // m₃ = m₂ * P = P³
    SET_M(3, 2);
    // m₄ = m₃ * P = P⁴
    SET_M(4, 3);
    // m₅ = m₄ * P = P⁵
    SET_M(5, 4);
    // m₆ = m₅ * P = P⁶
    SET_M(6, 5);
    // m₇ = m₆ * P = P⁷
    SET_M(7, 6);
    // m₈ = m₇ * P = P⁸
    SET_M(8, 7);
    // m₉ = m₈ * P = P⁹
    SET_M(9, 8);
    // m₁₀ = m₉ * P = P¹⁰
    SET_M(10, 9);
    // m₁₁ = m₁₀ * P = P¹¹
    SET_M(11, 10);

#undef SET_M

#define SET_EXP_GMP(d) mpz_class exp##d##_gmp = (m##d - 1) / mpz_class(6)

    // exp₁ = (m₁ - 1) / 6 = (P¹ - 1) / 6
    SET_EXP_GMP(1);
    // exp₂ = (m₂ - 1) / 6 = (P² - 1) / 6
    SET_EXP_GMP(2);
    // exp₃ = (m₃ - 1) / 6 = (P³ - 1) / 6
    SET_EXP_GMP(3);
    // exp₄ = (m₄ - 1) / 6 = (P⁴ - 1) / 6
    SET_EXP_GMP(4);
    // exp₅ = (m₅ - 1) / 6 = (P⁵ - 1) / 6
    SET_EXP_GMP(5);
    // exp₆ = (m₅ - 1) / 6 = (P⁶ - 1) / 6
    SET_EXP_GMP(6);
    // exp₇ = (m₆ - 1) / 6 = (P⁷ - 1) / 6
    SET_EXP_GMP(7);
    // exp₈ = (m₇ - 1) / 6 = (P⁸ - 1) / 6
    SET_EXP_GMP(8);
    // exp₉ = (m₈ - 1) / 6 = (P⁹ - 1) / 6
    SET_EXP_GMP(9);
    // exp₁₀ = (m₉ - 1) / 6 = (P¹⁰ - 1) / 6
    SET_EXP_GMP(10);
    // exp₁₁ = (m₁₀ - 1) / 6 = (P¹¹ - 1) / 6
    SET_EXP_GMP(11);

#undef SET_EXP_GMP

    // |kFrobeniusCoeffs[0]| = q^((P⁰ - 1) / 6)
    Config::kFrobeniusCoeffs[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF(d)                \
  BigInt<d * N> exp##d;                       \
  gmp::CopyLimbs(exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs[d] = BaseFieldConfig::kNonResidue.Pow(exp##d)
    // |kFrobeniusCoeffs[1]| = q^(exp₁) = q^((P¹ - 1) / 6)
    SET_FROBENIUS_COEFF(1);
    // |kFrobeniusCoeffs[2]| = q^(exp₂) = q^((P² - 1) / 6)
    SET_FROBENIUS_COEFF(2);
    // |kFrobeniusCoeffs[3]| = q^(exp₃) = q^((P³ - 1) / 6)
    SET_FROBENIUS_COEFF(3);
    // |kFrobeniusCoeffs[4]| = q^(exp₄) = q^((P⁴ - 1) / 6)
    SET_FROBENIUS_COEFF(4);
    // |kFrobeniusCoeffs[5]| = q^(exp₅) = q^((P⁵ - 1) / 6)
    SET_FROBENIUS_COEFF(5);
    // |kFrobeniusCoeffs[6]| = q^(exp₆) = q^((P⁶ - 1) / 6)
    SET_FROBENIUS_COEFF(6);
    // |kFrobeniusCoeffs[7]| = q^(exp₇) = q^((P⁷ - 1) / 6)
    SET_FROBENIUS_COEFF(7);
    // |kFrobeniusCoeffs[8]| = q^(exp₈) = q^((P⁸ - 1) / 6)
    SET_FROBENIUS_COEFF(8);
    // |kFrobeniusCoeffs[9]| = q^(exp₉) = q^((P⁹ - 1) / 6)
    SET_FROBENIUS_COEFF(9);
    // |kFrobeniusCoeffs[10]| = q^(exp₁₀) = q^((P¹⁰ - 1) / 6)
    SET_FROBENIUS_COEFF(10);
    // |kFrobeniusCoeffs[11]| = q^(exp₁₁) = q^((P¹¹ - 1) / 6)
    SET_FROBENIUS_COEFF(11);

#undef SET_FROBENIUS_COEFF
  }

  // CyclotomicMultiplicativeSubgroup methods
  Fp12 FastCyclotomicSquare() const {
    Fp12 ret;
    DoFastCyclotomicSquare(*this, ret);
    return ret;
  }

  Fp12& FastCyclotomicSquareInPlace() {
    DoFastCyclotomicSquare(*this, *this);
    return *this;
  }

  // Return α = (α₀', α₁', α₂', α₃', α₄', α₅'), such that
  // α = (α₀ + α₁x + α₂x² + (α₃ + α₄x + α₅x²)y) * (β₀ + β₃y + β₄xy)
  Fp12& MulInPlaceBy034(const Fp2& beta0, const Fp2& beta3, const Fp2& beta4) {
    // clang-format off
    // α = (α₀ + α₁x + α₂x² + (α₃ + α₄x + α₅x²)y) * (β₀ + β₃y + β₄xy)
    //   = (α₀β₀ + α₃β₃p + α₅β₄pq) + (α₁β₀ + α₃β₄p + α₄β₃p)x + (α₂β₀ + α₄β₄p + α₅β₃p)x² +
    //     (α₀β₃ + α₂β₄q + α₃β₀ + (α₀β₄ + α₁β₃ + α₄β₀)x + (α₁β₄ + α₂β₃ + α₅β₀)x²)y
    //   = (α₃β₃ + α₅β₄q + (α₃β₄ + α₄β₃)x + (α₄β₄ + α₅β₃)x²)p + α₀β₀ + α₁β₀x + α₂β₀x²
    //     (α₀β₃ + α₂β₄q + α₃β₀ + (α₀β₄ + α₁β₃ + α₄β₀)x + (α₁β₄ + α₂β₃ + α₅β₀)x²)y,
    //      where p = y² and q = x³
    // clang-format on

    // a = α₀β₀ + α₁β₀x + α₂β₀x²
    Fp6 a = this->c0_ * beta0;
    // b = (α₃ + α₄x + α₅x²) * (β₃ + β₄x)
    //   = (α₃β₃ + α₅β₄q) + (α₃β₄ + α₄β₃)x + (α₄β₄ + α₅β₃)x², where q = x³
    Fp6 b = this->c1_;
    b.MulInPlaceBy01(beta3, beta4);

    // c1 = (α₀ + α₃) + (α₁ + α₄)x + (α₂ + α₅)x²
    this->c1_ += this->c0_;
    // c1 = ((α₀ + α₃) + (α₁ + α₄)x + (α₂ + α₅)x²) * (β₀ + β₃ + β₄x)
    //    = ((α₀ + α₃) * (β₀ + β₃) + (α₂ + α₅)β₄q) +
    //      ((α₁ + α₄) * (β₀ + β₃) + (α₀ + α₃)β₄)x +
    //      ((α₂ + α₅) * (β₀ + β₃) + (α₁ + α₄)β₄)x², where q = x³
    //    = (α₀β₀ + α₀β₃ + α₃β₀ + α₃β₃ + α₂β₄q + α₅β₄q) +
    //      (α₁β₀ + α₁β₃ + α₄β₀ + α₄β₃ + α₀β₄ + α₃β₄)x +
    //      (α₂β₀ + α₂β₃ + α₅β₀ + α₅β₃ + α₁β₄ + α₄β₄)x², where q = x³
    //    = (α₀β₀ + α₀β₃ + α₂β₄q + α₃β₀ + α₃β₃ + α₅β₄q) +
    //      (α₀β₄ + α₁β₀ + α₁β₃ + α₃β₄ + α₄β₀ + α₄β₃)x +
    //      (α₁β₄ + α₂β₀ + α₂β₃ + α₄β₄ + α₅β₀ + α₅β₃)x², where q = x³
    this->c1_.MulInPlaceBy01(beta0 + beta3, beta4);
    // c1 = (α₀β₃ + α₂β₄q + α₃β₀ + α₃β₃ + α₅β₄q) +
    //      (α₀β₄ + α₁β₃ + α₃β₄ + α₄β₀ + α₄β₃)x +
    //      (α₁β₄ + α₂β₃ + α₄β₄ + α₅β₀ + α₅β₃)x², where q = x³
    this->c1_ -= a;
    // c1 = (α₀β₃ + α₂β₄q + α₃β₀) +
    //      (α₀β₄ + α₁β₃ + α₄β₀)x +
    //      (α₁β₄ + α₂β₃ + α₅β₀)x², where q = x³
    this->c1_ -= b;
    // c0 = ((α₃β₃ + α₅β₄q) + (α₄β₃ + α₃β₄)x + (α₅β₃ + α₄β₄)x²)p +
    //      α₀β₀ + α₁β₀x + α₂β₀x², where p = y² and q = x³
    this->c0_ = Config::MulByNonResidue(b);
    this->c0_ += a;
    return *this;
  }

  // Return α = (α₀', α₁', α₂', α₃', α₄', α₅'), such that
  // α = (α₀ + α₁x + α₂x² + (α₃ + α₄x + α₅x²)y) * (β₀ + β₁x + β₄xy)
  Fp12& MulInPlaceBy014(const Fp2& beta0, const Fp2& beta1, const Fp2& beta4) {
    // clang-format off
    // α = (α₀ + α₁x + α₂x² + (α₃ + α₄x + α₅x²)y) * (β₀ + β₁x + β₄xy)
    //   = (α₀β₀ + α₂β₁q + α₅β₄pq) + (α₀β₁ + α₁β₀ + α₃β₄p)x + (α₁β₁ + α₂β₀ + α₄β₄p)x² +
    //     (α₂β₄q + α₃β₀ + α₅β₁q + (α₀β₄ + α₃β₁ + α₄β₀)x + (α₁β₄ + α₄β₁ + α₅β₀)x²)y
    //   = (α₅β₄q + α₃β₄x + α₄β₄x²)p + (α₀β₀ + α₂β₁q) + (α₀β₁ + α₁β₀)x + (α₁β₁ + α₂β₀)x² +
    //     (α₂β₄q + α₃β₀ + α₅β₁q + (α₀β₄ + α₃β₁ + α₄β₀)x + (α₁β₄ + α₄β₁ + α₅β₀)x²)y,
    //      where p = y² and q = x³
    // clang-format on

    // c0 = (α₅β₄q + α₃β₄x + α₄β₄x²)p +
    //      (α₀β₀ + α₂β₁q) + (α₀β₁ + α₁β₀)x + (α₁β₁ + α₂β₀)x²,
    //      where p = y² and q = x³
    // c1 = (α₃β₀ + (α₂β₄ + α₅β₁)q) +
    //      (α₀β₄ + α₃β₁ + α₄β₀)x +
    //      (α₁β₄ + α₄β₁ + α₅β₀)x², where q = x³

    // a = (α₀ + α₁x + α₂x²) * (β₀ + β₁x)
    //   = (α₀β₀ + α₂β₁q) + (α₀β₁ + α₁β₀)x + (α₁β₁ + α₂β₀)x², where q = x³
    Fp6 a = this->c0_;
    a.MulInPlaceBy01(beta0, beta1);
    // b = (α₃ + α₄x + α₅x²) * β₄x
    //   = α₅β₄q + α₃β₄x + α₄β₄x²
    Fp6 b = this->c1_;
    b.MulInPlaceBy1(beta4);

    // c1 = (α₀ + α₃) + (α₁ + α₄)x + (α₂ + α₅)x²
    this->c1_ += this->c0_;
    // c1 = ((α₀ + α₃) + (α₁ + α₄)x + (α₂ + α₅)x²) * (β₀ + (β₁ + β₄)x)
    //    = ((α₀ + α₃) * β₀ + (α₂ + α₅)(β₁ + β₄)q) +
    //      ((α₁ + α₄) * β₀ + (α₀ + α₃)(β₁ + β₄))x +
    //      ((α₂ + α₅) * β₀ + (α₁ + α₄)(β₁ + β₄))x², where q = x³
    //    = (α₀β₀ + α₃β₀ + (α₂β₁ + α₂β₄ + α₅β₁ + α₅β₄)q) +
    //      (α₁β₀ + α₄β₀ + α₀β₁ + α₀β₄ + α₃β₁ + α₃β₄)x +
    //      (α₂β₀ + α₅β₀ + α₁β₁ + α₁β₄ + α₄β₁ + α₄β₄)x², where q = x³
    //    = (α₀β₀ + α₂β₁q + α₂β₄q + α₃β₀ + α₅β₁q + α₅β₄q) +
    //      (α₀β₁ + α₀β₄ + α₁β₀ + α₃β₁ + α₃β₄ + α₄β₀)x +
    //      (α₁β₁ + α₁β₄ + α₂β₀ + α₄β₁ + α₄β₄ + α₅β₀)x², where q = x³
    this->c1_.MulInPlaceBy01(beta0, beta1 + beta4);
    // c1 = (α₂β₄q + α₃β₀ + α₅β₁q + α₅β₄q) +
    //      (α₀β₄ + α₃β₁ + α₃β₄ + α₄β₀)x +
    //      (α₁β₄ + α₄β₁ + α₄β₄ + α₅β₀)x², where q = x³
    this->c1_ -= a;
    // c1 = (α₂β₄q + α₃β₀ + α₅β₁q) +
    //      (α₀β₄ + α₃β₁ + α₄β₀)x +
    //      (α₁β₄ + α₄β₁ + α₅β₀)x², where q = x³
    this->c1_ -= b;
    // c0 = (α₅β₄q + α₃β₄x + α₄β₄x²)p +
    //      (α₀β₀ + α₂β₁q) + (α₁β₀ + α₀β₁)x + (α₂β₀ + α₁β₁)x²,
    //      where p = y² and q = x³
    this->c0_ = Config::MulByNonResidue(b);
    this->c0_ += a;
    return *this;
  }

  static void DoFastCyclotomicSquare(const Fp12& a, Fp12& b) {
    // Faster Squaring in the Cyclotomic Subgroup of Sixth Degree Extensions
    // - Robert Granger and Michael Scott

    if constexpr (BasePrimeField::Config::kModulusModSixIsOne) {
      const Fp2& a0 = a.c0_.c0_;
      const Fp2& a1 = a.c0_.c1_;
      const Fp2& a2 = a.c0_.c2_;
      const Fp2& a3 = a.c1_.c0_;
      const Fp2& a4 = a.c1_.c1_;
      const Fp2& a5 = a.c1_.c2_;

      // a² = (α₀ + α₄x)² = α₀² + 2α₀α₄x + α₄²x²
      //                  = α₀² + α₄²q + 2α₀α₄x (where q = x²)
      //                  = t₀ + t₁x
      Fp2 tmp = a0 * a4;
      // t₀ = (α₀ + α₄) * (α₀ + α₄q) - α₀α₄ - α₀α₄x
      //    = α₀² + α₄²q
      Fp2 t0 = (a0 + a4) * (a0 + Fp6::Config::MulByNonResidue(a4)) - tmp -
               Fp6::Config::MulByNonResidue(tmp);
      // t₁ = 2α₀α₄
      Fp2 t1 = tmp.Double();

      // b² = (α₃ + α₂x)² = α₃² + 2α₂α₃x + α₂²x²
      //                  = α₃² + α₂²q + 2α₂α₃x (where q = x²)
      //                  = t₂ + t₃x
      tmp = a3 * a2;
      // t₂ = (α₃ + α₂) * (α₃ + α₂q) - α₂α₃ - α₂α₃x
      //    = α₃² + α₂²q
      Fp2 t2 = (a3 + a2) * (a3 + Fp6::Config::MulByNonResidue(a2)) - tmp -
               Fp6::Config::MulByNonResidue(tmp);
      // t₃ = 2α₂α₃
      Fp2 t3 = tmp.Double();

      // c² = (α₁ + α₅x)² = α₁² + 2α₁α₅x + α₅²x²
      //                  = α₁² + α₅²q + 2α₁α₅x (where q = x²)
      //                  = t₄ + t₅x
      tmp = a1 * a5;
      // t₄ = (α₁ + α₅) * (α₁ + α₅q) - α₁α₅ - α₁α₅x
      //    = α₁² + α₅²q
      Fp2 t4 = (a1 + a5) * (a1 + Fp6::Config::MulByNonResidue(a5)) - tmp -
               Fp6::Config::MulByNonResidue(tmp);
      // t₅ = 2α₁α₅
      Fp2 t5 = tmp.Double();

      Fp2& z0 = (&a == &b) ? b.c0_.c0_ : b.c0_.c0_ = a.c0_.c0_;
      Fp2& z4 = (&a == &b) ? b.c0_.c1_ : b.c0_.c1_ = a.c0_.c1_;
      Fp2& z3 = (&a == &b) ? b.c0_.c2_ : b.c0_.c2_ = a.c0_.c2_;
      Fp2& z2 = (&a == &b) ? b.c1_.c0_ : b.c1_.c0_ = a.c1_.c0_;
      Fp2& z1 = (&a == &b) ? b.c1_.c1_ : b.c1_.c1_ = a.c1_.c1_;
      Fp2& z5 = (&a == &b) ? b.c1_.c2_ : b.c1_.c2_ = a.c1_.c2_;

      // for A

      // z₀ = 3 * t₀ - 2 * z₀
      //    = 2 * (t₀ - z₀) + t₀
      z0 = t0 - z0;
      z0.DoubleInPlace();
      z0 += t0;

      // z₁ = 3 * t₁ + 2 * z₁
      //    = 2 * (t₁ + z₁) + t₁
      z1 = t1 + z1;
      z1.DoubleInPlace();
      z1 += t1;

      // for B

      // z₂ = 3 * (q * t₅) + 2 * z₂
      //    = 2 * (z₂ + q * t₅) + q * t₅
      tmp = Fp6::Config::MulByNonResidue(t5);
      z2 += tmp;
      z2.DoubleInPlace();
      z2 += tmp;

      // z₃ = 3 * t₄ - 2 * z₃
      //    = 2 * (t₄ - z₃) + t₄
      z3 = t4 - z3;
      z3.DoubleInPlace();
      z3 += t4;

      // for C

      // z₄ = 3 * t₂ - 2 * z₄
      //    = 2 * (t₂ - z₄) + t₂
      z4 = t2 - z4;
      z4.DoubleInPlace();
      z4 += t2;

      // z₅ = 3 * t₃ + 2 * z₅
      //    = 2 * (t₃ + z₅) + t₃
      z5 += t3;
      z5.DoubleInPlace();
      z5 += t3;
    } else {
      if (&a == &b) {
        b.SquareInPlace();
      } else {
        b = a.Square();
      }
    }
  }
};

template <typename Config>
struct ExtensionFieldTraits<Fp12<Config>> {
  constexpr static uint32_t kDegreeOverBaseField = Config::kDegreeOverBaseField;
  constexpr static uint32_t kDegreeOverBasePrimeField = 12;

  using BaseField = typename Fp12<Config>::BaseField;
  using BasePrimeField = typename Fp12<Config>::BasePrimeField;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP12_H_
