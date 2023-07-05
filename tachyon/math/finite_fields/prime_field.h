#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field_fallback.h"
#include "tachyon/math/finite_fields/prime_field_gmp.h"

namespace tachyon {
namespace math {

template <typename F, size_t MODULUS_BITS>
#if defined(TACHYON_GMP_BACKEND)
using PrimeField = PrimeFieldGmp<F, MODULUS_BITS>;
#else  // !defined(TACHYON_GMP_BACKEND)
using PrimeField = PrimeFieldFallback<F, MODULUS_BITS>;
#endif

class TACHYON_EXPORT Fp7 : public PrimeField<Fp7, 3> {
 public:
  using value_type = PrimeField<Fp7, 3>::value_type;

  using PrimeField<Fp7, 3>::PrimeField;

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_
