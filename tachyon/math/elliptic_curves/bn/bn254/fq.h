#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {
namespace bn254 {

class TACHYON_EXPORT Fq : public PrimeField<Fq, 254> {
 public:
  using value_type = PrimeField<Fq, 254>::value_type;

  using PrimeField<Fq, 254>::PrimeField;

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
