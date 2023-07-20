#ifndef TACHYON_MATH_FINITE_FIELDS_MODULUS_H_
#define TACHYON_MATH_FINITE_FIELDS_MODULUS_H_

#include "absl/numeric/internal/bits.h"

#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/bit_traits.h"

#if ARCH_CPU_BIG_ENDIAN
#define FOR_FROM_BIGGEST(start, end) for (size_t i = start; i < end; ++i)
#else  // ARCH_CPU_LITTLE_ENDIAN
#define FOR_FROM_BIGGEST(start, end) \
  for (size_t i = end - 1; i != static_cast<size_t>(start - 1); --i)
#endif

#if ARCH_CPU_BIG_ENDIAN
#define FOR_FROM_SMALLEST(start, end) \
  for (size_t i = end - 1; i != static_cast<size_t>(start - 1); --i)
#else  // ARCH_CPU_LITTLE_ENDIAN
#define FOR_FROM_SMALLEST(start, end) for (size_t i = start; i < end; ++i)
#endif

namespace tachyon {
namespace math {
namespace internal {

template <size_t N>
struct RBuffer {
  uint64_t limbs[N] = {
      0,
  };
  uint64_t value = 1;
};

template <size_t N>
struct R2Buffer {
  uint64_t lo[N] = {
      0,
  };
  uint64_t hi[N] = {
      0,
  };
  uint64_t value = 1;
};

}  // namespace internal

template <size_t N>
class BitTraits<internal::RBuffer<N>> {
 public:
  static constexpr bool kIsDynamic = false;

  static constexpr size_t GetNumBits(const internal::RBuffer<N>& r_buffer) {
    return N * 64 +
           (64 - absl::numeric_internal::CountLeadingZeroes64(r_buffer.value));
  }

  static constexpr bool TestBit(const internal::RBuffer<N>& r_buffer,
                                size_t index) {
    size_t limb_index = index >> 6;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = static_cast<uint64_t>(1) << bit_index;
    uint64_t value =
        limb_index == N ? r_buffer.value : r_buffer.limbs[limb_index];
    return (value & bit_index_value) == bit_index_value;
  }
};

template <size_t N>
class BitTraits<internal::R2Buffer<N>> {
 public:
  static constexpr bool kIsDynamic = false;

  static constexpr size_t GetNumBits(const internal::R2Buffer<N>& r2_buffer) {
    return 2 * N * 64 +
           (64 - absl::numeric_internal::CountLeadingZeroes64(r2_buffer.value));
  }

  static constexpr bool TestBit(const internal::R2Buffer<N>& r2_buffer,
                                size_t index) {
    size_t limb_index = index >> 6;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = static_cast<uint64_t>(1) << bit_index;
    uint64_t value = limb_index == 2 * N
                         ? r2_buffer.value
                         : (limb_index >= N ? r2_buffer.hi[limb_index - N]
                                            : r2_buffer.lo[limb_index]);
    return (value & bit_index_value) == bit_index_value;
  }
};

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

  static constexpr BigInt<N> MontgomeryR(const BigInt<N>& modulus) {
    internal::RBuffer<N> two_pow_n_times_64;
    return Mod(two_pow_n_times_64, modulus);
  }

  static constexpr BigInt<N> MontgomeryR2(const BigInt<N>& modulus) {
    internal::R2Buffer<N> two_pow_n_times_64_square;
    return Mod(two_pow_n_times_64_square, modulus);
  }

  // Compute -M^{-1} mod 2^B.
  template <typename T, size_t B = 8 * sizeof(T),
            std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  static constexpr T Inverse(const BigInt<N>& modulus) {
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

 private:
  template <typename T>
  static constexpr BigInt<N> Mod(const T& buffer, const BigInt<N>& modulus) {
    // Stupid slow base-2 long division taken from
    // https://en.wikipedia.org/wiki/Division_algorithm
    CHECK(!modulus.IsZero());
    BigInt<N> remainder;
    size_t bits = BitTraits<T>::GetNumBits(buffer);
    uint64_t carry = 0;
    uint64_t& smallest_bit = remainder.limbs[BigInt<N>::kSmallestLimbIdx];
    FOR_FROM_BIGGEST(0, bits) {
      remainder.MulBy2InPlace(carry);
      smallest_bit |= BitTraits<T>::TestBit(buffer, i);
      if (remainder >= modulus || carry) {
        uint64_t borrow = 0;
        remainder.SubInPlace(modulus, borrow);
        CHECK_EQ(borrow, carry);
      }
    }
    return remainder;
  }
};

#undef FOR_FROM_BIGGEST
#undef FOR_FROM_SMALLEST

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_MODULUS_H_
