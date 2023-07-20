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

  constexpr static uint64_t ExtensionDegree() { return 1; }

  static void Init();
};

using Fq = PrimeField<FqConfig>;
#if defined(TACHYON_GMP_BACKEND)
using FqGmp = PrimeFieldGmp<FqConfig>;
#endif  // defined(TACHYON_GMP_BACKEND)
#if TACHYON_CUDA
using FqCuda = PrimeFieldMontCuda<FqConfig>;
#endif  // TACHYON_CUDA

}  // namespace bn254
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
