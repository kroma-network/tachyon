#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FR_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FR_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

class TACHYON_EXPORT FrConfig {
 public:
  constexpr static size_t kModulusBits = 255;

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();

  static PrimeField<FrConfig>& Modulus();
};

using Fr = PrimeField<FrConfig>;

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FR_H_
