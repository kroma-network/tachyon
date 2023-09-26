// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FP2_H_
#define TACHYON_MATH_FINITE_FIELDS_FP2_H_

#include "tachyon/math/finite_fields/quadratic_extension_field.h"

namespace tachyon::math {

template <typename Config>
class Fp2 : public QuadraticExtensionField<Fp2<Config>> {
 public:
  using BaseField = typename Config::BaseField;

  using QuadraticExtensionField<Fp2<Config>>::QuadraticExtensionField;

  static_assert(BaseField::ExtensionDegree() == 1);

  constexpr static uint64_t kDegreeOverBasePrimeField = 2;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FP2_H_
