#include "tachyon/math/finite_fields/prime_field_util.h"

namespace tachyon::math {

// The integer s such that |n| = |k|^s * t for some odd integer t.
uint32_t ComputeAdicity(uint32_t k, mpz_class n) {
  uint32_t adicity = 0;
  while (n > 1) {
    if (n % k == 0) {
      adicity += 1;
      n /= k;
    } else {
      break;
    }
  }
  return adicity;
}

// The integer t such that |n| = |k|^s * t for some odd integer t.
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
