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

  using Fp = BasePrimeField;
  using Fp3 = BaseField;

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

  // Return α = {α₀', α₁', α₂', α₃', α₄', α₅'}, such that
  // α = (α₀ + α₁x + α₂x² + (α₃ + α₄x + α₅x²)y) * (β₀ + β₃y + β₄xy)
  Fp6& MulInPlaceBy034(const Fp& beta0, const Fp& beta3, const Fp& beta4) {
    // α = (α₀ + α₁x + α₂x² + (α₃ + α₄x + α₅x²)y) * (β₀ + β₃y + β₄xy)
    //   = (α₀β₀ + α₄β₄q + α₅β₃q) + <- I am not clear here
    //     (α₁β₀ + α₃β₃ + α₅β₄q)x + <- I am not clear here
    //     (α₂β₀ + α₃β₄ + α₄β₃)x² + <- I am not clear here
    //     (α₀β₃ + α₂β₄q + α₃β₀)y +
    //     (α₀β₄ + α₁β₃ + α₄β₀)xy +
    //     (α₁β₄ + α₂β₃ + α₅β₀)x²y, where q is a cubic non residue.
    // NOTE(chokobole): This equation above works when assuming y² = x holds.

    // z0 = α₀
    Fp z0 = this->c0_.c0_;
    // z1 = α₁
    Fp z1 = this->c0_.c1_;
    // z2 = α₂
    Fp z2 = this->c0_.c2_;
    // z3 = α₃
    Fp z3 = this->c1_.c0_;
    // z4 = α₄
    Fp z4 = this->c1_.c1_;
    // z5 = α₅
    Fp z5 = this->c1_.c2_;

    // x0 = β₀
    Fp x0 = beta0;
    // x3 = β₃
    Fp x3 = beta3;
    // x4 = β₄
    Fp x4 = beta4;

    // tmp1 = β₃q
    Fp tmp1 = Fp3::Config::MulByNonResidue(x3);
    // tmp2 = β₄q
    Fp tmp2 = Fp3::Config::MulByNonResidue(x4);

    // α₀' = α₀β₀ + α₄β₄q + α₅β₃q
    this->c0_.c0_ = (z0 * x0) + (z4 * tmp2) + (z5 * tmp1);
    // α₁' = α₁β₀ + α₃β₃ + α₅β₄q
    this->c0_.c1_ = (z1 * x0) + (z3 * x3) + (z5 * tmp2);
    // α₂' = α₂β₀ + α₃β₄ + α₄β₃
    this->c0_.c2_ = (z2 * x0) + (z3 * x4) + (z4 * x3);
    // α₃' = α₀β₃ + α₂β₄q + α₃β₀
    this->c1_.c0_ = (z0 * x3) + (z2 * tmp2) + (z3 * x0);
    // α₄' = α₀β₄ + α₁β₃ + α₄β₀
    this->c1_.c1_ = (z0 * x4) + (z1 * x3) + (z4 * x0);
    // α₅' = α₁β₄ + α₂β₃ + α₅β₀
    this->c1_.c2_ = (z1 * x4) + (z2 * x3) + (z5 * x0);
    return *this;
  }

  // Return α = {α₀', α₁', α₂', α₃', α₄', α₅'}, such that
  // α = (α₀ + α₁x + α₂x² + (α₃ + α₄x + α₅x²)y) * (β₀ + β₁x + β₄xy)
  Fp6& MulInPlaceBy014(const Fp& beta0, const Fp& beta1, const Fp& beta4) {
    // α = (α₀ + α₁x + α₂x² + (α₃ + α₄x + α₅x²)y) * (β₀ + β₁x + β₄xy)
    //   = (α₀β₀ + α₂β₁q + α₄β₄q) + <- I am not clear here
    //     (α₀β₁ + α₁β₀ + α₅β₄q)x + <- I am not clear here
    //     (α₁β₁ + α₂β₀ + α₃β₄)x² + <- I am not clear here
    //     (α₂β₄q + α₃β₀ + α₅β₁q)y +
    //     (α₀β₄ + α₃β₁ + α₄β₀)xy +
    //     (α₁β₄ + α₄β₁ + α₅β₀)x²y, where q is a cubic non residue.
    // NOTE(chokobole): This equation above works when assuming y² = x holds.

    // z0 = α₀
    Fp z0 = this->c0_.c0_;
    // z1 = α₁
    Fp z1 = this->c0_.c1_;
    // z2 = α₂
    Fp z2 = this->c0_.c2_;
    // z3 = α₃
    Fp z3 = this->c1_.c0_;
    // z4 = α₄
    Fp z4 = this->c1_.c1_;
    // z5 = α₅
    Fp z5 = this->c1_.c2_;

    // x0 = β₀
    Fp x0 = beta0;
    // x1 = β₁
    Fp x1 = beta1;
    // x4 = β₄
    Fp x4 = beta4;

    // tmp1 = β₁q
    Fp tmp1 = Fp3::Config::MulByNonResidue(x1);
    // tmp2 = β₄q
    Fp tmp2 = Fp3::Config::MulByNonResidue(x4);

    // α₀' = α₀β₀ + α₂β₁q + α₄β₄q
    this->c0_.c0_ = (z0 * x0) + (z2 * tmp1) + (z4 * tmp2);
    // α₁' = α₀β₁ + α₁β₀ + α₅β₄q
    this->c0_.c1_ = (z0 * x1) + (z1 * x0) + (z5 * tmp2);
    // α₂' = α₁β₁ + α₂β₀ + α₃β₄
    this->c0_.c2_ = (z1 * x1) + (z2 * x0) + (z3 * x4);
    // α₃' = α₂β₄q + α₃β₀ + α₅β₁q
    this->c1_.c0_ = (z2 * tmp2) + (z3 * x0) + (z5 * tmp1);
    // α₄' = α₀β₄ + α₃β₁ + α₄β₀
    this->c1_.c1_ = (z0 * x4) + (z3 * x1) + (z4 * x0);
    // α₅' = α₁β₄ + α₄β₁ + α₅β₀
    this->c1_.c2_ = (z1 * x4) + (z4 * x1) + (z5 * x0);
    return *this;
  }
};

template <typename Config>
class Fp6<Config, std::enable_if_t<Config::kDegreeOverBaseField == 3>> final
    : public CubicExtensionField<Fp6<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using FrobeniusCoefficient = typename Config::FrobeniusCoefficient;

  using Fp2 = BaseField;

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
  Fp6& MulInPlaceBy1(const Fp2& beta1) {
    // α = (α₀ + α₁x + α₂x²) * β₁x
    //   = α₂β₁q + α₀β₁x + α₁β₁x², where q is a cubic non residue.

    // t0 = α₂β₁
    Fp2 t0 = this->c2_ * beta1;

    // c2 = α₁β₁
    this->c2_ = this->c1_ * beta1;
    // c1 = α₀β₁
    this->c1_ = this->c0_ * beta1;
    // c0 = α₂β₁q
    this->c0_ = Config::MulByNonResidue(t0);
    return *this;
  }

  // Return α = {α₀', α₁', α₂'}, such that α = (α₀ + α₁x + α₂x²) * (β₀ + β₁x)
  Fp6& MulInPlaceBy01(const Fp2& beta0, const Fp2& beta1) {
    // α = (α₀ + α₁x + α₂x²) * (β₀ + β₁x)
    //   = α₀β₀ + α₂β₁q + (α₀β₁ + α₁β₀)x + (α₂β₀ + α₁β₁)x²,
    //     where q is a cubic non residue.

    // The naive approach you need to multiply 6 times, but this code is
    // optimized to multiply 5 times.

    // a_a = α₀β₀
    Fp2 a_a = this->c0_ * beta0;
    // b_b = α₁β₁
    Fp2 b_b = this->c1_ * beta1;

    Fp2 t0;
    {
      // tmp = α₁ + α₂
      Fp2 tmp = this->c1_ + this->c2_;

      // t0 = (α₁ + α₂)β₁ = α₁β₁ + α₂β₁
      t0 = tmp * beta1;
      // t0 = α₂β₁
      t0 -= b_b;
      // t0 = α₂β₁q
      t0 = Config::MulByNonResidue(t0);
      // t0 = α₀β₀ + α₂β₁q
      t0 += a_a;
    }

    // t1 = β₀ + β₁
    Fp2 t1 = beta0 + beta1;
    {
      // tmp = α₀ + α₁
      Fp2 tmp = this->c0_ + this->c1_;

      // t1 = (α₀ + α₁)(β₀ + β₁) = α₀β₀ + α₀β₁ + α₁β₀ + α₁β₁
      t1 *= tmp;
      // t1 = α₀β₁ + α₁β₀ + α₁β₁
      t1 -= a_a;
      // t1 = α₀β₁ + α₁β₀
      t1 -= b_b;
    }

    // t2 = β₀
    Fp2 t2;
    {
      // tmp = α₀ + α₂
      Fp2 tmp = this->c0_ + this->c2_;

      // t2 = (α₀ + α₂)β₀ = α₀β₀ + α₂β₀
      t2 = tmp * beta0;
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
