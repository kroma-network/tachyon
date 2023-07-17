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
  // clang-format off
  // Dec: 52435875175126190479447740508185965837690552500527637822603658699938581184513
  // Hex: 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
  // clang-format on
  constexpr static uint64_t kModulus[4] = {
      UINT64_C(18446744069414584321),
      UINT64_C(6034159408538082302),
      UINT64_C(3691218898639771653),
      UINT64_C(8353516859464449352),
  };

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Fr = PrimeField<FrConfig>;

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FR_H_
