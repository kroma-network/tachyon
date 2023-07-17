#ifndef TACHYON_MATH_FINITE_FIELDS_MODULUS_H_
#define TACHYON_MATH_FINITE_FIELDS_MODULUS_H_

#include "tachyon/math/base/big_int.h"

namespace tachyon {
namespace math {

template <size_t N>
class Modulus {
 public:
  // Can we use the no-carry optimization for multiplication
  // outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
  //
  // This optimization applies if
  // (a) `modulus[biggest_limb_idx] < max(uint64_t) >> 1`, and
  // (b) the bits of the modulus are not all 1.
  static constexpr bool CanUseNoCarryMulOptimization(const BigInt<N>& modulus) {
    uint64_t biggest_limb = modulus[BigInt<N>::kBiggestLimbIdx];
    bool top_bit_is_zero = biggest_limb >> 63 == 0;
    bool all_remain_bits_are_one =
        biggest_limb == std::numeric_limits<uint64_t>::max() >> 1;
#if ARCH_CPU_BIG_ENDIAN
    for (size_t i = 0; i < N - 1; ++i) {
#else  // ARCH_CPU_LITTLE_ENDIAN
    for (size_t i = 1; i < N; ++i) {
#endif
      all_remain_bits_are_one &=
          modulus[N - i - 1] == std::numeric_limits<uint64_t>::max();
    }
    return top_bit_is_zero && !all_remain_bits_are_one;
  }

  // Does the modulus have a spare unused bit?
  //
  // This condition applies if
  // (a) `modulus[biggest_limb_idx] >> 63 == 0`
  static constexpr bool HasSparseBit(const BigInt<N>& modulus) {
    uint64_t biggest_limb = modulus[BigInt<N>::kBiggestLimbIdx];
    return biggest_limb >> 63 == 0;
  }
};

#undef FOR_FROM_BIGGEST
#undef FOR_FROM_SMALLEST

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_MODULUS_H_
