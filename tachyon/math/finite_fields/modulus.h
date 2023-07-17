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
  // (a) `modulus[N - 1] < max(uint64_t) >> 1`, and
  // (b) the bits of the modulus are not all 1.
  static constexpr bool CanUseNoCarryMulOptimization(const BigInt<N>& modulus) {
    bool top_bit_is_zero = modulus[N - 1] >> 63 == 0;
    bool all_remain_bits_are_one =
        modulus[N - 1] == std::numeric_limits<uint64_t>::max() >> 1;
    for (size_t i = 1; i < N; ++i) {
      all_remain_bits_are_one &=
          modulus[N - i - 1] == std::numeric_limits<uint64_t>::max();
    }
    return top_bit_is_zero && !all_remain_bits_are_one;
  }

  // Does the modulus have a spare unused bit?
  //
  // This condition applies if
  // (a) `modulus[N - 1] >> 63 == 0`
  static constexpr bool HasSparseBit(const BigInt<N>& modulus) {
    return modulus[N - 1] >> 63 == 0;
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_MODULUS_H_
