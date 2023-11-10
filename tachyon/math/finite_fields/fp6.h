// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP6_H_
#define TACHYON_MATH_FINITE_FIELDS_FP6_H_

#include <utility>

#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/cubic_extension_field.h"
#include "tachyon/math/finite_fields/quadratic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp6<Config, std::enable_if_t<Config::kDegreeOverBaseField == 2>> final
    : public QuadraticExtensionField<Fp6<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using CpuField = Fp6<Config>;
  // TODO(chokobole): Implements Fp6Gpu
  using GpuField = Fp6<Config>;

  using QuadraticExtensionField<Fp6<Config>>::QuadraticExtensionField;

  static_assert(BaseField::ExtensionDegree() == 3);

  constexpr static uint64_t kDegreeOverBasePrimeField = 6;

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

#undef SET_EXP_GMP

    // kFrobeniusCoeffs[0] = q^((P⁰ - 1) / 6) = 1
    Config::kFrobeniusCoeffs[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF(d)                \
  BigInt<d * N> exp##d;                       \
  gmp::CopyLimbs(exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs[d] = BaseFieldConfig::kNonResidue.Pow(exp##d)

    // kFrobeniusCoeffs[1] = q^(exp₁) = q^((P¹ - 1) / 6) = ω
    SET_FROBENIUS_COEFF(1);
    // kFrobeniusCoeffs[2] = q^(exp₂) = q^((P² - 1) / 6)
    SET_FROBENIUS_COEFF(2);
    // kFrobeniusCoeffs[3] = q^(exp₃) = q^((P³ - 1) / 6)
    SET_FROBENIUS_COEFF(3);
    // kFrobeniusCoeffs[4] = q^(exp₄) = q^((P⁴ - 1) / 6)
    SET_FROBENIUS_COEFF(4);
    // kFrobeniusCoeffs[5] = q^(exp₅) = q^((P⁵ - 1) / 6)
    SET_FROBENIUS_COEFF(5);

#undef SET_FROBENIUS_COEFF
  }
};

template <typename Config>
class Fp6<Config, std::enable_if_t<Config::kDegreeOverBaseField == 3>> final
    : public CubicExtensionField<Fp6<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using Fp2Ty = BaseField;

  using CpuField = Fp6<Config>;
  // TODO(chokobole): Implements Fp6Gpu
  using GpuField = Fp6<Config>;

  using CubicExtensionField<Fp6<Config>>::CubicExtensionField;

  static_assert(BaseField::ExtensionDegree() == 2);

  constexpr static uint64_t kDegreeOverBasePrimeField = 6;

  static void Init() {
    Config::Init();
    // x³ = q = Config::kNonResidue

    // αᴾ = (α₀ + α₁x + α₂x²)ᴾ
    //    = α₀ᴾ + α₁ᴾxᴾ + α₂ᴾx²ᴾ
    //    = ᾱ₀ + ᾱ₁xᴾ + ᾱx²ᴾ <- conjugate
    //    = ᾱ₀ + ᾱ₁xᴾ⁻¹x + ᾱx²ᴾ⁻²x²
    //    = ᾱ₀ + ᾱ₁(x³)^((P - 1) / 3) * x + ᾱ(x³)^(2 * (P - 1) / 3) * x²
    //    = ᾱ₀ + ᾱ₁ωx + ᾱω²x², where ω is a cubic root of unity.

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

#undef SET_M

#define SET_EXP_GMP(d) mpz_class exp##d##_gmp = (m##d - 1) / mpz_class(3)

    // exp₁ = (m₁ - 1) / 3 = (P¹ - 1) / 3
    SET_EXP_GMP(1);
    // exp₂ = (m₂ - 1) / 3 = (P² - 1) / 3
    SET_EXP_GMP(2);
    // exp₃ = (m₃ - 1) / 3 = (P³ - 1) / 3
    SET_EXP_GMP(3);
    // exp₄ = (m₄ - 1) / 3 = (P⁴ - 1) / 3
    SET_EXP_GMP(4);
    // exp₅ = (m₅ - 1) / 3 = (P⁵ - 1) / 3
    SET_EXP_GMP(5);

#undef SET_EXP_GMP

    // kFrobeniusCoeffs[0] = q^((P⁰ - 1) / 3)
    Config::kFrobeniusCoeffs[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF(d)                \
  BigInt<d * N> exp##d;                       \
  gmp::CopyLimbs(exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs[d] = Config::kNonResidue.Pow(exp##d)

    // kFrobeniusCoeffs[1] = q^(exp₁) = q^((P¹ - 1) / 3)
    SET_FROBENIUS_COEFF(1);
    // kFrobeniusCoeffs[2] = q^(exp₂) = q^((P² - 1) / 3)
    SET_FROBENIUS_COEFF(2);
    // kFrobeniusCoeffs[3] = q^(exp₃) = q^((P³ - 1) / 3)
    SET_FROBENIUS_COEFF(3);
    // kFrobeniusCoeffs[4] = q^(exp₄) = q^((P⁴ - 1) / 3)
    SET_FROBENIUS_COEFF(4);
    // kFrobeniusCoeffs[5] = q^(exp₅) = q^((P⁵ - 1) / 3)
    SET_FROBENIUS_COEFF(5);

#undef SET_FROBENIUS_COEFF

    // kFrobeniusCoeffs2[0] = q^((P⁰ - 1) / 3)
    Config::kFrobeniusCoeffs2[0] = FrobeniusCoefficient::One();
#define SET_FROBENIUS_COEFF2(d)                              \
  gmp::CopyLimbs(mpz_class(2) * exp##d##_gmp, exp##d.limbs); \
  Config::kFrobeniusCoeffs2[d] = Config::kNonResidue.Pow(exp##d)

    // kFrobeniusCoeffs2[1] = q^(2 * exp₁) = q^(2 * (P¹ - 1) / 3)
    SET_FROBENIUS_COEFF2(1);
    // kFrobeniusCoeffs2[2] = q^(2 * exp₂) = q^(2 * (P² - 1) / 3)
    SET_FROBENIUS_COEFF2(2);
    // kFrobeniusCoeffs2[3] = q^(2 * exp₃) = q^(2 * (P³ - 1) / 3)
    SET_FROBENIUS_COEFF2(3);
    // kFrobeniusCoeffs2[4] = q^(2 * exp₄) = q^(2 * (P⁴ - 1) / 3)
    SET_FROBENIUS_COEFF2(4);
    // kFrobeniusCoeffs2[5] = q^(2 * exp₅) = q^(2 * (P⁵ - 1) / 3)
    SET_FROBENIUS_COEFF2(5);

#undef SET_FROBENIUS_COEFF2
  }

  // Return α = {α₀', α₁', α₂'}, such that α = (α₀ + α₁x + α₂x²) * β₁x
  Fp6& MulInPlaceBy1(const Fp2Ty& beta1) {
    // α = (α₀ + α₁x + α₂x²) * β₁x
    //   = α₂β₁q + α₀β₁x + α₁β₁x², where q is a cubic non residue.

    // t0 = α₂
    Fp2Ty t0 = this->c2_;
    // t0 = α₂β₁
    t0 *= beta1;
    // t0 = α₂β₁q
    t0 = Config::MulByNonResidue(t0);

    // t1 = α₀
    Fp2Ty t1 = this->c0_;
    // t1 = α₀β₁
    t1 *= beta1;

    // t2 = α₁
    Fp2Ty t2 = this->c1_;
    // t2 = α₁β₁
    t2 *= beta1;

    // c0 = α₂β₁q
    this->c0_ = std::move(t0);
    // c1 = α₀β₁
    this->c1_ = std::move(t1);
    // c2 = α₁β₁
    this->c2_ = std::move(t2);
    return *this;
  }

  // Return α = {α₀', α₁', α₂'}, such that α = (α₀ + α₁x + α₂x²) * (β₀ + β₁x)
  Fp6& MulInPlaceBy01(const Fp2Ty& beta0, const Fp2Ty& beta1) {
    // α = (α₀ + α₁x + α₂x²) * (β₀ + β₁x)
    //   = α₀β₀ + α₂β₁q + (α₀β₁ + α₁β₀)x + (α₂β₀ + α₁β₁)x²,
    //     where q is a cubic non residue.

    // The naive approach you need to multiply 6 times, but this code is
    // optimized to multiply 5 times.

    // a_a = α₀β₀
    Fp2Ty a_a = this->c0_;
    a_a *= beta0;
    // b_b = α₁β₁
    Fp2Ty b_b = this->c1_;
    b_b *= beta1;

    // t0 = β₁
    Fp2Ty t0 = beta1;
    {
      // tmp = α₁ + α₂
      Fp2Ty tmp = this->c1_;
      tmp += this->c2_;

      // t0 = (α₁ + α₂)β₁ = α₁β₁ + α₂β₁
      t0 *= tmp;
      // t0 = α₂β₁
      t0 -= b_b;
      // t0 = α₂β₁q
      t0 = Config::MulByNonResidue(t0);
      // t0 = α₀β₀ + α₂β₁q
      t0 += a_a;
    }

    // t1 = β₀
    Fp2Ty t1 = beta0;
    // t1 = β₀ + β₁
    t1 += beta1;
    {
      // tmp = α₀ + α₁
      Fp2Ty tmp = this->c0_;
      tmp += this->c1_;

      // t1 = (α₀ + α₁)(β₀ + β₁) = α₀β₀ + α₀β₁ + α₁β₀ + α₁β₁
      t1 *= tmp;
      // t1 = α₀β₁ + α₁β₀ + α₁β₁
      t1 -= a_a;
      // t1 = α₀β₁ + α₁β₀
      t1 -= b_b;
    }

    // t2 = β₀
    Fp2Ty t2 = beta0;
    {
      // tmp = α₀ + α₂
      Fp2Ty tmp = this->c0_;
      tmp += this->c2_;

      // t2 = (α₀ + α₂)β₀ = α₀β₀ + α₂β₀
      t2 *= tmp;
      // t2 = α₂β₀
      t2 -= a_a;
      // t2 = α₂β₀ + α₁β₁
      t2 += b_b;
    }

    // c0 = α₀β₀ + α₂β₁q
    this->c0_ = std::move(t0);
    // c1 = α₀β₁ + α₁β₀
    this->c1_ = std::move(t1);
    // c2 = α₂β₀ + α₁β₁
    this->c2_ = std::move(t2);
    return *this;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP6_H_
