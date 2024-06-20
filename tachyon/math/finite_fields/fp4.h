// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP4_H_
#define TACHYON_MATH_FINITE_FIELDS_FP4_H_

#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/quadratic_extension_field.h"
#include "tachyon/math/finite_fields/quartic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp4<Config, std::enable_if_t<Config::kDegreeOverBaseField == 2>> final
    : public QuadraticExtensionField<Fp4<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using CpuField = Fp4<Config>;
  // TODO(chokobole): Implement Fp4Gpu
  using GpuField = Fp4<Config>;

  using QuadraticExtensionField<Fp4<Config>>::QuadraticExtensionField;

  static_assert(Config::kDegreeOverBaseField == 2);
  static_assert(BaseField::ExtensionDegree() == 2);

  constexpr static uint64_t kDegreeOverBasePrimeField = 4;

  static void Init() {
    using BaseFieldConfig = typename BaseField::Config;
    // x⁴ = q = |BaseFieldConfig::kNonResidue|

    Config::Init();

    // αᴾ = (α₀ + α₁x)ᴾ
    //    = α₀ᴾ + α₁ᴾxᴾ
    //    = ᾱ₀ + ᾱ₁xᴾ <- conjugate
    //    = ᾱ₀ + ᾱ₁xᴾ⁻¹x
    //    = ᾱ₀ + ᾱ₁(x⁴)^((p - 1) / 4) * x
    //    = ᾱ₀ - ᾱ₁ωx, where ω is a quartic root of unity.

    constexpr uint64_t N = BasePrimeField::kLimbNums;
    // m₁ = P
    mpz_class m1;
    if constexpr (BasePrimeField::Config::kModulusBits <= 32) {
      m1 = mpz_class(BasePrimeField::Config::kModulus);
    } else {
      gmp::WriteLimbs(BasePrimeField::Config::kModulus.limbs, N, &m1);
    }

#define SET_M(d, d_prev) mpz_class m##d = m##d_prev * m1

    // m₂ = m₁ * P = P²
    SET_M(2, 1);
    // m₃ = m₂ * P = P³
    SET_M(3, 2);

#undef SET_M

#define SET_EXP_GMP(d) mpz_class exp##d##_gmp = (m##d - 1) / mpz_class(4)

    // exp₁ = (m₁ - 1) / 4 = (P¹ - 1) / 4
    SET_EXP_GMP(1);
    // exp₂ = (m₂ - 1) / 4 = (P² - 1) / 4
    SET_EXP_GMP(2);
    // exp₃ = (m₃ - 1) / 4 = (P³ - 1) / 4
    SET_EXP_GMP(3);

#undef SET_EXP_GMP

    // |kFrobeniusCoeffs[0]| = q^((P⁰ - 1) / 4) = 1
    Config::kFrobeniusCoeffs[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF(d)                \
  BigInt<d * N> exp##d;                       \
  gmp::CopyLimbs(exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs[d] = BaseFieldConfig::kNonResidue.Pow(exp##d)

    // |kFrobeniusCoeffs[1]| = q^(exp₁) = q^((P¹ - 1) / 4) = ω
    SET_FROBENIUS_COEFF(1);
    // |kFrobeniusCoeffs[2]| = q^(exp₂) = q^((P² - 1) / 4)
    SET_FROBENIUS_COEFF(2);
    // |kFrobeniusCoeffs[3]| = q^(exp₃) = q^((P³ - 1) / 4)
    SET_FROBENIUS_COEFF(3);

#undef SET_FROBENIUS_COEFF
  }
};

template <typename Config>
class Fp4<Config, std::enable_if_t<Config::kDegreeOverBaseField == 4>> final
    : public QuarticExtensionField<Fp4<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using CpuField = Fp4<Config>;
  // TODO(chokobole): Implement Fp4Gpu
  using GpuField = Fp4<Config>;

  using QuarticExtensionField<Fp4<Config>>::QuarticExtensionField;

  static_assert(Config::kDegreeOverBaseField == 4);
  static_assert(BaseField::ExtensionDegree() == 1);

  constexpr static uint64_t kDegreeOverBasePrimeField = 4;

  static void Init() {
    Config::Init();
    // x⁴ = q = |Config::kNonResidue|

    // αᴾ = (α₀ + α₁x + α₂x² + α₃x³)ᴾ
    //    = α₀ᴾ + α₁ᴾxᴾ + α₂ᴾx²ᴾ + α₃ᴾx³ᴾ
    //    = α₀ + α₁xᴾ + α₂x²ᴾ + α₃x³ᴾ <- Fermat's little theorem
    //    = α₀ + α₁xᴾ⁻¹x + α₂x²ᴾ⁻²x² + α₃x³ᴾ⁻³x³
    //    = α₀ + α₁(x⁴)^((P - 1) / 4) * x + α₂(x⁴)^(2 * (P - 1) / 4) * x² +
    //      α₃(x⁴)^(3 * (P - 1) / 4) * x³
    //    = α₀ + α₁ωx + α₂ω²x² + α₃ω³x³, where ω is a quartic root of unity.

    constexpr uint64_t N = BasePrimeField::kLimbNums;
    // m₁ = P
    mpz_class m1;
    if constexpr (BasePrimeField::Config::kModulusBits <= 32) {
      m1 = mpz_class(BasePrimeField::Config::kModulus);
    } else {
      gmp::WriteLimbs(BasePrimeField::Config::kModulus.limbs, N, &m1);
    }

#define SET_M(d, d_prev) mpz_class m##d = m##d_prev * m1

    // m₂ = m₁ * P = P²
    SET_M(2, 1);
    // m₃ = m₂ * P = P³
    SET_M(3, 2);

#undef SET_M

#define SET_EXP_GMP(d) mpz_class exp##d##_gmp = (m##d - 1) / mpz_class(4)

    // exp₁ = (m₁ - 1) / 4 = (P¹ - 1) / 4
    SET_EXP_GMP(1);
    // exp₂ = (m₂ - 1) / 4 = (P² - 1) / 4
    SET_EXP_GMP(2);
    // exp₃ = (m₃ - 1) / 4 = (P³ - 1) / 4
    SET_EXP_GMP(3);

#undef SET_EXP_GMP

    // |kFrobeniusCoeffs[0]| = q^((P⁰ - 1) / 4) = 1
    Config::kFrobeniusCoeffs[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF(d)                \
  BigInt<d * N> exp##d;                       \
  gmp::CopyLimbs(exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs[d] = Config::kNonResidue.Pow(exp##d)

    // |kFrobeniusCoeffs[1]| = q^(exp₁) = q^((P¹ - 1) / 4) = ω
    SET_FROBENIUS_COEFF(1);
    // |kFrobeniusCoeffs[2]| = q^(exp₂) = q^((P² - 1) / 4)
    SET_FROBENIUS_COEFF(2);
    // |kFrobeniusCoeffs[3]| = q^(exp₃) = q^((P³ - 1) / 4)
    SET_FROBENIUS_COEFF(3);

#undef SET_FROBENIUS_COEFF

    // |kFrobeniusCoeffs2[0]| = q^(2 * (P⁰ - 1) / 4) = 1
    Config::kFrobeniusCoeffs2[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF2(d)                              \
  gmp::CopyLimbs(mpz_class(2) * exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs2[d] = Config::kNonResidue.Pow(exp##d)

    // |kFrobeniusCoeffs2[1]| = q^(2 * exp₁) = q^(2 * (P¹ - 1) / 4) = ω²
    SET_FROBENIUS_COEFF2(1);
    // |kFrobeniusCoeffs2[2]| = q^(2 * exp₂) = q^(2 * (P² - 1) / 4)
    SET_FROBENIUS_COEFF2(2);
    // |kFrobeniusCoeffs2[3]| = q^(2 * exp₃) = q^(2 * (P³ - 1) / 4)
    SET_FROBENIUS_COEFF2(3);

#undef SET_FROBENIUS_COEFF2

    // |kFrobeniusCoeffs3[0]| = q^(3 * (P⁰ - 1) / 4) = 1
    Config::kFrobeniusCoeffs3[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF3(d)                              \
  gmp::CopyLimbs(mpz_class(3) * exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs3[d] = Config::kNonResidue.Pow(exp##d)

    // |kFrobeniusCoeffs3[1]| = q^(3 * exp₁) = q^(3 * (P¹ - 1) / 4) = ω³
    SET_FROBENIUS_COEFF3(1);
    // |kFrobeniusCoeffs3[2]| = q^(3 * exp₂) = q^(3 * (P² - 1) / 4)
    SET_FROBENIUS_COEFF3(2);
    // |kFrobeniusCoeffs3[3]| = q^(3 * exp₃) = q^(3 * (P³ - 1) / 4)
    SET_FROBENIUS_COEFF3(3);

#undef SET_FROBENIUS_COEFF3
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP4_H_
