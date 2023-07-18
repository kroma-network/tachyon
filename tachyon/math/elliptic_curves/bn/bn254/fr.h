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
  // clang-format off
  // Dec: 21888242871839275222246405745257275088548364400416034343698204186575808495617
  // Hex: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
  // clang-format on
  constexpr static BigInt<4> kModulus = BigInt<4>({
      UINT64_C(13401866920200346009),
      UINT64_C(16891761104669281089),
      UINT64_C(10551491231982245282),
      UINT64_C(348699826680297066),
  });

  constexpr static bool kCanUseNoCarryMulOptimization = true;
  constexpr static bool kModulusHasSparseBit = true;
  constexpr static BigInt<4> kMontgomeryR = BigInt<4>({
      UINT64_C(4079194950544968940),
      UINT64_C(7072138095255860902),
      UINT64_C(4724778148209793768),
      UINT64_C(314353086334104154),
  });
  constexpr static BigInt<4> kMontgomeryR2 = BigInt<4>({
      UINT64_C(13819146929316105287),
      UINT64_C(335017290899456567),
      UINT64_C(15308605460182742983),
      UINT64_C(149885893874197625),
  });
  constexpr static uint64_t kInverse = UINT64_C(14599321983923869015);

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Fr = PrimeField<FrConfig>;
#if defined(TACHYON_GMP_BACKEND)
using FrGmp = PrimeFieldGmp<FrConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
