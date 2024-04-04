// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#include "tachyon/math/finite_fields/prime_field_util.h"

namespace tachyon::math {

// The integer s such that |n| = |k|ˢ * t for some odd integer t.
uint32_t ComputeAdicity(uint32_t k, mpz_class n) {
  uint32_t adicity = 0;
  while (n > 1) {
    if (n % k == 0) {
      ++adicity;
      n /= k;
    } else {
      break;
    }
  }
  return adicity;
}

// The integer t such that |n| = |k|ˢ * t for some odd integer t.
mpz_class ComputeTrace(size_t k, mpz_class n) {
  mpz_class trace = 0;
  while (n > 1) {
    if (n % k == 0) {
      n /= k;
    } else {
      trace = n;
      break;
    }
  }
  return trace;
}

}  // namespace tachyon::math
