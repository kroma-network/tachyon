// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP3_H_
#define TACHYON_MATH_FINITE_FIELDS_FP3_H_

#include "tachyon/math/finite_fields/cubic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp3 : public CubicExtensionField<Fp3<Config>> {
 public:
  using BaseField = typename Config::BaseField;

  using CpuField = Fp3<Config>;
  // TODO(chokobole): Implements Fp3Gpu
  using GpuField = Fp3<Config>;

  using CubicExtensionField<Fp3<Config>>::CubicExtensionField;

  static_assert(Config::kDegreeOverBaseField == 3);
  static_assert(BaseField::ExtensionDegree() == 1);

  constexpr static uint64_t kDegreeOverBasePrimeField = 3;

  static void Init() { Config::Init(); }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP3_H_
