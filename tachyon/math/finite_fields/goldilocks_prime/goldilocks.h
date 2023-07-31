#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_H_

#if defined(TACHYON_GOLDILOCKS_BACKEND)
#include "tachyon/math/finite_fields/goldilocks_prime/prime_field_goldilocks.h"
#else
#include "tachyon/math/finite_fields/prime_field.h"
#endif

namespace tachyon::math {

class TACHYON_EXPORT GoldilocksConfig {
 public:
#if defined(TACHYON_GOLDILOCKS_BACKEND)
  constexpr static bool kIsGoldilocks = true;
#else
  constexpr static bool kIsSpecialPrime = false;
#endif

  constexpr static size_t kModulusBits = 64;
  // 2^64 - 2^32 + 1
  // Dec: 18446744069414584321
  // Hex: 0xffffffff00000001
  constexpr static BigInt<1> kModulus =
      BigInt<1>(UINT64_C(18446744069414584321));
  constexpr static BigInt<1> kOne = BigInt<1>(UINT64_C(4294967295));

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Goldilocks = PrimeField<GoldilocksConfig>;
#if defined(TACHYON_GMP_BACKEND)
using GoldilocksGmp = PrimeFieldGmp<GoldilocksConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_GOLDILOCKS_H_
