// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP4_H_
#define TACHYON_MATH_FINITE_FIELDS_FP4_H_

#include "tachyon/math/finite_fields/quadratic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp4 final : public QuadraticExtensionField<Fp4<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;

  using CpuField = Fp4<Config>;
  // TODO(chokobole): Implements Fp4Gpu
  using GpuField = Fp4<Config>;

  using QuadraticExtensionField<Fp4<Config>>::QuadraticExtensionField;

  static_assert(Config::kDegreeOverBaseField == 2);
  static_assert(BaseField::ExtensionDegree() == 2);

  constexpr static uint64_t kDegreeOverBasePrimeField = 4;

  static void Init() { Config::Init(); }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP4_H_
