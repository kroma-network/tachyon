#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {
namespace bn254 {

class TACHYON_EXPORT FqConfig {
 public:
  constexpr static size_t kModulusBits = 254;
  // clang-format off
  // Dec: 21888242871839275222246405745257275088696311157297823662689037894645226208583
  // Hex: 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
  // clang-format on
  constexpr static BigInt<4> kModulus = BigInt<4>({
      UINT64_C(4332616871279656263),
      UINT64_C(10917124144477883021),
      UINT64_C(13281191951274694749),
      UINT64_C(3486998266802970665),
  });

  constexpr static bool kCanUseNoCarryMulOptimization = true;
  constexpr static bool kModulusHasSparseBit = true;
  constexpr static BigInt<4> kMontgomeryR = BigInt<4>({
      UINT64_C(15230403791020821917),
      UINT64_C(754611498739239741),
      UINT64_C(7381016538464732716),
      UINT64_C(1011752739694698287),
  });
  constexpr static BigInt<4> kMontgomeryR2 = BigInt<4>({
      UINT64_C(17522657719365597833),
      UINT64_C(13107472804851548667),
      UINT64_C(5164255478447964150),
      UINT64_C(493319470278259999),
  });
  constexpr static uint64_t kInverse = UINT64_C(9786893198990664585);

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Fq = PrimeField<FqConfig>;
#if defined(TACHYON_GMP_BACKEND)
using FqGmp = PrimeFieldGmp<FqConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
