#ifndef TACHYON_MATH_FINITE_FIELDS_MODULUS_H_
#define TACHYON_MATH_FINITE_FIELDS_MODULUS_H_

#include "tachyon/math/base/big_int.h"

namespace tachyon::math {

template <size_t N>
class Modulus {
 public:
  // Can we use the no-carry optimization for multiplication
  // outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
  //
  // This optimization applies if
  // (a) `modulus[biggest_limb_idx] < max(uint64_t) >> 1`, and
  // (b) the bits of the modulus are not all 1.
  constexpr static bool CanUseNoCarryMulOptimization(const BigInt<N>& modulus) {
    uint64_t biggest_limb = modulus[BigInt<N>::kBiggestLimbIdx];
    bool top_bit_is_zero = biggest_limb >> 63 == 0;
    bool all_remain_bits_are_one =
        biggest_limb == std::numeric_limits<uint64_t>::max() >> 1;
    FOR_BUT_SMALLEST(i, N) {
      all_remain_bits_are_one &=
          modulus[N - i - 1] == std::numeric_limits<uint64_t>::max();
    }
    return top_bit_is_zero && !all_remain_bits_are_one;
  }

  // Does the modulus have a spare unused bit?
  //
  // This condition applies if
  // (a) `modulus[biggest_limb_idx] >> 63 == 0`
  constexpr static bool HasSpareBit(const BigInt<N>& modulus) {
    uint64_t biggest_limb = modulus[BigInt<N>::kBiggestLimbIdx];
    return biggest_limb >> 63 == 0;
  }

  constexpr static BigInt<N> MontgomeryR(const BigInt<N>& modulus) {
    BigInt<N + 1> two_pow_n_times_64;
    two_pow_n_times_64.biggest_limb() = static_cast<uint64_t>(1);
    return (two_pow_n_times_64 % modulus.template Extend<N + 1>())
        .template Shrink<N>();
  }

  constexpr static BigInt<N> MontgomeryR2(const BigInt<N>& modulus) {
    BigInt<2 * N + 1> two_pow_n_times_64_square;
    two_pow_n_times_64_square.biggest_limb() = static_cast<uint64_t>(1);
    return (two_pow_n_times_64_square % modulus.template Extend<2 * N + 1>())
        .template Shrink<N>();
  }

  // Compute -M^{-1} mod 2^B.
  template <typename T, size_t B = 8 * sizeof(T),
            std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr static T Inverse(const BigInt<N>& modulus) {
    // We compute this as follows.
    // First, modulus mod 2^B is just the lower B bits of modulus.
    // Hence modulus mod 2^B = modulus[0] mod 2^B.
    //
    // Next, computing the inverse mod 2^B involves exponentiating by
    // the multiplicative group order, which is euler_totient(2^B) - 1.
    // Now, euler_totient(2^B) = 1 << (B - 1), and so
    // euler_totient(2^B) - 1 = (1 << (B - 1)) - 1 = 1111111... ((B - 1)
    // digits). We compute this powering via standard square and multiply.
    T inv = 1;
    for (size_t i = 0; i < (B - 1); ++i) {
      // Square
      inv *= inv;
      // Multiply
      inv *= static_cast<T>(modulus[0]);
    };
    return -inv;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_MODULUS_H_
