#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {
namespace bn254 {

class TACHYON_EXPORT FqConfig {
 public:
  constexpr static size_t MODULUS_BITS = 254;

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();

  static PrimeField<FqConfig>& Modulus();
};

using Fq = PrimeField<FqConfig>;

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
