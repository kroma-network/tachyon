#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field_fallback.h"
#include "tachyon/math/finite_fields/prime_field_gmp.h"

namespace tachyon {
namespace math {

template <typename Config>
#if defined(TACHYON_GMP_BACKEND)
using PrimeField = PrimeFieldGmp<Config>;
#else  // !defined(TACHYON_GMP_BACKEND)
using PrimeField = PrimeFieldFallback<Config>;
#endif

class TACHYON_EXPORT GF7Config {
 public:
  constexpr static size_t MODULUS_BITS = 3;

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();

  static PrimeField<GF7Config>& Modulus();
};

using GF7 = PrimeField<GF7Config>;

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_
