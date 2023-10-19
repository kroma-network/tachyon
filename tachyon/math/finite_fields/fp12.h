// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP12_H_
#define TACHYON_MATH_FINITE_FIELDS_FP12_H_

#include "tachyon/math/finite_fields/quadratic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp12 final : public QuadraticExtensionField<Fp12<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;

  using CpuField = Fp12<Config>;
  // TODO(chokobole): Implements Fp12Gpu
  using GpuField = Fp12<Config>;

  using QuadraticExtensionField<Fp12<Config>>::QuadraticExtensionField;

  static_assert(Config::kDegreeOverBaseField == 2);
  static_assert(BaseField::ExtensionDegree() == 6);

  constexpr static uint64_t kDegreeOverBasePrimeField = 12;

  static void Init() { Config::Init(); }

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
