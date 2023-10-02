// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP4_H_
#define TACHYON_MATH_FINITE_FIELDS_FP4_H_

#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/quadratic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp4 final : public QuadraticExtensionField<Fp4<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using CpuField = Fp4<Config>;
  // TODO(chokobole): Implements Fp4Gpu
  using GpuField = Fp4<Config>;

  using QuadraticExtensionField<Fp4<Config>>::QuadraticExtensionField;

  static_assert(Config::kDegreeOverBaseField == 2);
  static_assert(BaseField::ExtensionDegree() == 2);

  constexpr static uint64_t kDegreeOverBasePrimeField = 4;

  static void Init() {
    using BaseFieldConfig = typename BaseField::Config;
    // x⁴ = q = BaseFieldConfig::kNonResidue

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
    gmp::WriteLimbs(BasePrimeField::Config::kModulus.limbs, N, &m1);

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

    // kFrobeniusCoeffs[0] = q^((P⁰ - 1) / 4) = 1
    Config::kFrobeniusCoeffs[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF(d)                \
  BigInt<d * N> exp##d;                       \
  gmp::CopyLimbs(exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs[d] = BaseFieldConfig::kNonResidue.Pow(exp##d)

    // kFrobeniusCoeffs[1] = q^(exp₁) = q^((P¹ - 1) / 4) = ω
    SET_FROBENIUS_COEFF(1);
    // kFrobeniusCoeffs[2] = q^(exp₂) = q^((P² - 1) / 4)
    SET_FROBENIUS_COEFF(2);
    // kFrobeniusCoeffs[3] = q^(exp₃) = q^((P³ - 1) / 4)
    SET_FROBENIUS_COEFF(3);

#undef SET_FROBENIUS_COEFF
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP4_H_
