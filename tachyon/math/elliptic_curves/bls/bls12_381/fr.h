#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FR_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FR_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"
#if defined(TACHYON_GMP_BACKEND)
#include "tachyon/math/finite_fields/prime_field_gmp.h"
#endif  // defined(TACHYON_GMP_BACKEND)

namespace tachyon::math {
namespace bls12_381 {

class TACHYON_EXPORT FrConfig {
 public:
  constexpr static bool kIsSpecialPrime = false;

  constexpr static size_t kModulusBits = 255;
  // clang-format off
  // Parameters are from https://electriccoin.co/blog/new-snark-curve/
  // Dec: 52435875175126190479447740508185965837690552500527637822603658699938581184513
  // Hex: 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
  // clang-format on
  constexpr static BigInt<4> kModulus = BigInt<4>({
      UINT64_C(18446744069414584321),
      UINT64_C(6034159408538082302),
      UINT64_C(3691218898639771653),
      UINT64_C(8353516859464449352),
  });
  constexpr static BigInt<4> kOne = BigInt<4>({
      UINT64_C(8589934590),
      UINT64_C(6378425256633387010),
      UINT64_C(11064306276430008309),
      UINT64_C(1739710354780652911),
  });

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Fr = PrimeField<FrConfig>;
#if defined(TACHYON_GMP_BACKEND)
using FrGmp = PrimeFieldGmp<FrConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace bls12_381
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FR_H_
