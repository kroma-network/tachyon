#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {
namespace bn254 {

class TACHYON_EXPORT FrConfig {
 public:
  constexpr static size_t kModulusBits = 254;

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();

  static PrimeField<FrConfig>& Modulus();
};

using Fr = PrimeField<FrConfig>;

}  // namespace bn254

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
