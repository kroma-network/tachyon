#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_H_

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

class TACHYON_EXPORT FqConfig {
 public:
  constexpr static size_t kModulusBits = 381;
  // clang-format off
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

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Fq = PrimeFieldGmp<FqConfig>;

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BLS_BLS12_381_FQ_H_
