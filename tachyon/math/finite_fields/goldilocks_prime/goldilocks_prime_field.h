#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_PRIME_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_PRIME_FIELD_H_

#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon::math {

class TACHYON_EXPORT GoldilocksConfig {
 public:
  constexpr static size_t kModulusBits = 64;
  // 2^64 - 2^32 + 1
  // Dec: 18446744069414584321
  // Hex: 0xffffffff00000001
  constexpr static BigInt<1> kModulus =
      BigInt<1>(UINT64_C(18446744069414584321));

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Goldilocks = PrimeField<GoldilocksConfig>;
#if defined(TACHYON_GMP_BACKEND)
using GoldilocksGmp = PrimeFieldGmp<GoldilocksConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_PRIME_FIELD_H_
