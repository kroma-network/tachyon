// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP3_H_
#define TACHYON_MATH_FINITE_FIELDS_FP3_H_

#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/cubic_extension_field.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"

namespace tachyon::math {

template <typename Config>
class Fp3 final : public CubicExtensionField<Fp3<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using CpuField = Fp3<Config>;
  // TODO(chokobole): Implement Fp3Gpu
  using GpuField = Fp3<Config>;

  using CubicExtensionField<Fp3<Config>>::CubicExtensionField;

  static_assert(Config::kDegreeOverBaseField == 3);
  static_assert(BaseField::ExtensionDegree() == 1);

  constexpr static uint32_t kDegreeOverBasePrimeField = 3;

  static void Init() {
    Config::Init();
    // x³ = q = |Config::kNonResidue|

    // αᴾ = (α₀ + α₁x + α₂x²)ᴾ
    //    = α₀ᴾ + α₁ᴾxᴾ + α₂ᴾx²ᴾ
    //    = α₀ + α₁xᴾ + α₂x²ᴾ <- Fermat's little theorem
    //    = α₀ + α₁xᴾ⁻¹x + α₂x²ᴾ⁻²x²
    //    = α₀ + α₁(x³)^((P - 1) / 3) * x + α₂(x³)^(2 * (P - 1) / 3) * x²
    //    = α₀ + α₁ωx + α₂ω²x², where ω is a cubic root of unity.

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

#undef SET_M

#define SET_EXP_GMP(d) mpz_class exp##d##_gmp = (m##d - 1) / mpz_class(3)

    // exp₁ = (m₁ - 1) / 3 = (P¹ - 1) / 3
    SET_EXP_GMP(1);
    // exp₂ = (m₂ - 1) / 3 = (P² - 1) / 3
    SET_EXP_GMP(2);

#undef SET_EXP_GMP

    // |kFrobeniusCoeffs[0]| = q^((P⁰ - 1) / 3) = 1
    Config::kFrobeniusCoeffs[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF(d)                \
  BigInt<d * N> exp##d;                       \
  gmp::CopyLimbs(exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs[d] = Config::kNonResidue.Pow(exp##d)

    // |kFrobeniusCoeffs[1]| = q^(exp₁) = q^((P¹ - 1) / 3) = ω
    SET_FROBENIUS_COEFF(1);
    // |kFrobeniusCoeffs[2]| = q^(exp₂) = q^((P² - 1) / 3)
    SET_FROBENIUS_COEFF(2);

#undef SET_FROBENIUS_COEFF

    // kFrobeniusCoeffs2[0] = q^(2 * (P⁰ - 1) / 3) = 1
    Config::kFrobeniusCoeffs2[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF2(d)                              \
  gmp::CopyLimbs(mpz_class(2) * exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs2[d] = Config::kNonResidue.Pow(exp##d)

    // kFrobeniusCoeffs2[1] = q^(2 * exp₁) = q^(2 * (P¹ - 1) / 3) = ω²
    SET_FROBENIUS_COEFF2(1);
    // kFrobeniusCoeffs2[2] = q^(2 * exp₂) = q^(2 * (P² - 1) / 3)
    SET_FROBENIUS_COEFF2(2);

#undef SET_FROBENIUS_COEFF2
  }
};

template <typename Config>
struct ExtensionFieldTraits<Fp3<Config>> {
  constexpr static uint32_t kDegreeOverBaseField = 3;
  constexpr static uint32_t kDegreeOverBasePrimeField = 3;

  using BaseField = typename Fp3<Config>::BaseField;
  using BasePrimeField = typename Fp3<Config>::BasePrimeField;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP3_H_
