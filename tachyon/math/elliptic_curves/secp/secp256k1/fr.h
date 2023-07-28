#ifndef TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FR_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FR_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon::math {
namespace secp256k1 {

class TACHYON_EXPORT FrConfig {
 public:
  constexpr static size_t kModulusBits = 256;
  // clang-format off
  // Parameters are from https://www.secg.org/sec2-v2.pdf#page=13
  // Dec: 115792089237316195423570985008687907852837564279074904382605163141518161494337
  // Hex: fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
  // clang-format on
  constexpr static BigInt<4> kModulus = BigInt<4>({
      UINT64_C(13822214165235122497),
      UINT64_C(13451932020343611451),
      UINT64_C(18446744073709551614),
      UINT64_C(18446744073709551615),
  });
  constexpr static BigInt<4> kOne = BigInt<4>({
      UINT64_C(4624529908474429119),
      UINT64_C(4994812053365940164),
      UINT64_C(1),
      UINT64_C(0),
  });

  static void Init();
};

using Fr = PrimeField<FrConfig>;
#if defined(TACHYON_GMP_BACKEND)
using FrGmp = PrimeFieldGmp<FrConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace secp256k1
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_SECP_SECP256K1_FR_H_
