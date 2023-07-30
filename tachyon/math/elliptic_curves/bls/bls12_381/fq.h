#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"
#if defined(TACHYON_GMP_BACKEND)
#include "tachyon/math/finite_fields/prime_field_gmp.h"
#endif  // defined(TACHYON_GMP_BACKEND)

namespace tachyon::math {
namespace bls12_381 {

class TACHYON_EXPORT FqConfig {
 public:
  constexpr static bool kIsSpecialPrime = false;

  constexpr static size_t kModulusBits = 381;
  // clang-format off
  // Parameters are from https://electriccoin.co/blog/new-snark-curve/
  // Dec: 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
  // Hex: 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
  // clang-format on
  constexpr static BigInt<6> kModulus = BigInt<6>({
      UINT64_C(13402431016077863595),
      UINT64_C(2210141511517208575),
      UINT64_C(7435674573564081700),
      UINT64_C(7239337960414712511),
      UINT64_C(5412103778470702295),
      UINT64_C(1873798617647539866),
  });
  constexpr static BigInt<6> kOne = BigInt<6>({
      UINT64_C(8505329371266088957),
      UINT64_C(17002214543764226050),
      UINT64_C(6865905132761471162),
      UINT64_C(8632934651105793861),
      UINT64_C(6631298214892334189),
      UINT64_C(1582556514881692819),
  });

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Fq = PrimeField<FqConfig>;
#if defined(TACHYON_GMP_BACKEND)
using FqGmp = PrimeFieldGmp<FqConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)

}  // namespace bls12_381
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_H_
