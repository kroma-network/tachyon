#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field_mont.h"

#if defined(TACHYON_GMP_BACKEND)
#include "tachyon/math/finite_fields/prime_field_gmp.h"
#endif  // defined(TACHYON_GMP_BACKEND)

namespace tachyon {
namespace math {

template <typename Config>
using PrimeField = PrimeFieldMont<Config>;

class TACHYON_EXPORT GF7Config {
 public:
  constexpr static size_t kModulusBits = 3;
  constexpr static BigInt<1> kModulus = BigInt<1>(7);

  constexpr static bool kCanUseNoCarryMulOptimization = true;
  constexpr static bool kModulusHasSparseBit = true;
  constexpr static BigInt<1> kMontgomeryR = BigInt<1>(2);
  constexpr static BigInt<1> kMontgomeryR2 = BigInt<1>(4);
  constexpr static uint64_t kInverse = UINT64_C(10540996613548315209);

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using GF7 = PrimeField<GF7Config>;
#if defined(TACHYON_GMP_BACKEND)
using GF7Gmp = PrimeFieldGmp<GF7Config>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_
