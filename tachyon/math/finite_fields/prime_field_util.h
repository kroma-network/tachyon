// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_UTIL_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_UTIL_H_

#include <stdint.h>

#include "third_party/gmp/include/gmpxx.h"

#include "tachyon/export.h"

namespace tachyon::math {

TACHYON_EXPORT uint32_t ComputeAdicity(uint32_t k, mpz_class n);

TACHYON_EXPORT mpz_class ComputeTrace(size_t k, mpz_class n);

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_UTIL_H_
