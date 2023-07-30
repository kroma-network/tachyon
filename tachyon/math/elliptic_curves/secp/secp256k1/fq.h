#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FQ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FQ_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"
#if defined(TACHYON_GMP_BACKEND)
#include "tachyon/math/finite_fields/prime_field_gmp.h"
#endif  // defined(TACHYON_GMP_BACKEND)

namespace tachyon::math {
namespace secp256k1 {

class TACHYON_EXPORT FqConfig {
 public:
  constexpr static bool kIsSpecialPrime = false;

  constexpr static size_t kModulusBits = 256;
  // clang-format off
  // Parameters are from https://www.secg.org/sec2-v2.pdf#page=13
  // Dec: 115792089237316195423570985008687907853269984665640564039457584007908834671663
  // Hex: fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
  // clang-format on
  constexpr static BigInt<4> kModulus = BigInt<4>({
      UINT64_C(18446744069414583343),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
  });
  constexpr static BigInt<4> kOne = BigInt<4>({
      UINT64_C(4294968273),
      UINT64_C(0),
      UINT64_C(0),
      UINT64_C(0),
  });

  static void Init();
};

using Fq = PrimeField<FqConfig>;
#if defined(TACHYON_GMP_BACKEND)
using FqGmp = PrimeFieldGmp<FqConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace secp256k1
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FQ_H_
