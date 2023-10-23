// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP12_H_
#define TACHYON_MATH_FINITE_FIELDS_FP12_H_

#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/quadratic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp12 final : public QuadraticExtensionField<Fp12<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using CpuField = Fp12<Config>;
  // TODO(chokobole): Implements Fp12Gpu
  using GpuField = Fp12<Config>;

  using QuadraticExtensionField<Fp12<Config>>::QuadraticExtensionField;

  static_assert(Config::kDegreeOverBaseField == 2);
  static_assert(BaseField::ExtensionDegree() == 6);

  constexpr static uint64_t kDegreeOverBasePrimeField = 12;

  static void Init() {
    using BaseFieldConfig = typename BaseField::Config;
    // x⁶ = q = BaseFieldConfig::kNonResidue

    Config::Init();

    // αᴾ = (α₀ + α₁x)ᴾ
    //    = α₀ᴾ + α₁ᴾxᴾ
    //    = α₀ᴾ + α₁ᴾxᴾ⁻¹x
    //    = α₀ᴾ + α₁ᴾ(x⁶)^((P - 1) / 6) * x
    //    = α₀ᴾ + α₁ᴾωx, where ω is a sextic root of unity.

    constexpr uint64_t N = BasePrimeField::kLimbNums;
    // m₁ = P
    mpz_class m1;
    gmp::WriteLimbs(BasePrimeField::Config::kModulus.limbs, N, &m1);

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

    // kFrobeniusCoeffs[0] = q^((P⁰ - 1) / 6)
    Config::kFrobeniusCoeffs[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF(d)                \
  BigInt<d * N> exp##d;                       \
  gmp::CopyLimbs(exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs[d] = BaseFieldConfig::kNonResidue.Pow(exp##d)
    // kFrobeniusCoeffs[1] = q^(exp₁) = q^((P¹ - 1) / 6)
    SET_FROBENIUS_COEFF(1);
    // kFrobeniusCoeffs[2] = q^(exp₂) = q^((P² - 1) / 6)
    SET_FROBENIUS_COEFF(2);
    // kFrobeniusCoeffs[3] = q^(exp₃) = q^((P³ - 1) / 6)
    SET_FROBENIUS_COEFF(3);
    // kFrobeniusCoeffs[4] = q^(exp₄) = q^((P⁴ - 1) / 6)
    SET_FROBENIUS_COEFF(4);
    // kFrobeniusCoeffs[5] = q^(exp₅) = q^((P⁵ - 1) / 6)
    SET_FROBENIUS_COEFF(5);
    // kFrobeniusCoeffs[6] = q^(exp₆) = q^((P⁶ - 1) / 6)
    SET_FROBENIUS_COEFF(6);
    // kFrobeniusCoeffs[7] = q^(exp₇) = q^((P⁷ - 1) / 6)
    SET_FROBENIUS_COEFF(7);
    // kFrobeniusCoeffs[8] = q^(exp₈) = q^((P⁸ - 1) / 6)
    SET_FROBENIUS_COEFF(8);
    // kFrobeniusCoeffs[9] = q^(exp₉) = q^((P⁹ - 1) / 6)
    SET_FROBENIUS_COEFF(9);
    // kFrobeniusCoeffs[10] = q^(exp₁₀) = q^((P¹⁰ - 1) / 6)
    SET_FROBENIUS_COEFF(10);
    // kFrobeniusCoeffs[11] = q^(exp₁₁) = q^((P¹¹ - 1) / 6)
    SET_FROBENIUS_COEFF(11);

#undef SET_FROBENIUS_COEFF
  }

  // CyclotomicMultiplicativeSubgroup methods
  Fp12& FastCyclotomicSquareInPlace() {
    // Faster Squaring in the Cyclotomic Subgroup of Sixth Degree Extensions
    // - Robert Granger and Michael Scott

    if constexpr (BasePrimeField::Config::kModulusModSixIsOne) {
      using Fp6Ty = BaseField;
      using Fp2Ty = typename Fp6Ty::BaseField;
      const Fp2Ty& a0 = this->c0_.c0_;
      const Fp2Ty& a1 = this->c0_.c1_;
      const Fp2Ty& a2 = this->c0_.c2_;
      const Fp2Ty& a3 = this->c1_.c0_;
      const Fp2Ty& a4 = this->c1_.c1_;
      const Fp2Ty& a5 = this->c1_.c2_;

      // a² = (α₀ + α₄x)² = α₀² + 2α₀α₄x + α₄²x²
      //                  = α₀² + α₄²q + 2α₀α₄x (where q = x²)
      //                  = t₀ + t₁x
      Fp2Ty tmp = a0 * a4;
      // t₀ = (α₀ + α₄) * (α₀ + α₄q) - α₀α₄ - α₀α₄x
      //    = α₀² + α₄²q
      Fp2Ty t0 = (a0 + a4) * (a0 + Fp6Ty::Config::MulByNonResidue(a4)) - tmp -
                 Fp6Ty::Config::MulByNonResidue(tmp);
      // t₁ = 2α₀α₄
      Fp2Ty t1 = tmp.Double();

      // b² = (α₃ + α₂x)² = α₃² + 2α₂α₃x + α₂²x²
      //                  = α₃² + α₂²q + 2α₂α₃x (where q = x²)
      //                  = t₂ + t₃x
      tmp = a3 * a2;
      // t₂ = (α₃ + α₂) * (α₃ + α₂q) - α₂α₃ - α₂α₃x
      //    = α₃² + α₂²q
      Fp2Ty t2 = (a3 + a2) * (a3 + Fp6Ty::Config::MulByNonResidue(a2)) - tmp -
                 Fp6Ty::Config::MulByNonResidue(tmp);
      // t₃ = 2α₂α₃
      Fp2Ty t3 = tmp.Double();

      // c² = (α₁ + α₅x)² = α₁² + 2α₁α₅x + α₅²x²
      //                  = α₁² + α₅²q + 2α₁α₅x (where q = x²)
      //                  = t₄ + t₅x
      tmp = a1 * a5;
      // t₄ = (α₁ + α₅) * (α₁ + α₅q) - α₁α₅ - α₁α₅x
      //    = α₁² + α₅²q
      Fp2Ty t4 = (a1 + a5) * (a1 + Fp6Ty::Config::MulByNonResidue(a5)) - tmp -
                 Fp6Ty::Config::MulByNonResidue(tmp);
      // t₅ = 2α₁α₅
      Fp2Ty t5 = tmp.Double();

      Fp2Ty& z0 = this->c0_.c0_;
      Fp2Ty& z4 = this->c0_.c1_;
      Fp2Ty& z3 = this->c0_.c2_;
      Fp2Ty& z2 = this->c1_.c0_;
      Fp2Ty& z1 = this->c1_.c1_;
      Fp2Ty& z5 = this->c1_.c2_;

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
      tmp = Fp6Ty::Config::MulByNonResidue(t5);
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

      return *this;
    } else {
      return this->SquareInPlace();
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP12_H_
