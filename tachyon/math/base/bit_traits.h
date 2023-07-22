#ifndef TACHYON_MATH_BASE_BIT_TRAITS_H_
#define TACHYON_MATH_BASE_BIT_TRAITS_H_

#include "tachyon/math/base/big_int.h"

namespace tachyon {
namespace math {

template <typename T, typename SFINAE = void>
class BitTraits;

template <size_t N>
class BitTraits<BigInt<N>> {
 public:
  constexpr static bool kIsDynamic = false;

  constexpr static size_t GetNumBits(const BigInt<N>& _) { return N * 64; }

  constexpr static bool TestBit(const BigInt<N>& bigint, size_t index) {
    size_t limb_index = index >> 6;
    if (limb_index >= N) return false;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = static_cast<uint64_t>(1) << bit_index;
    return (bigint[limb_index] & bit_index_value) == bit_index_value;
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_BIT_TRAITS_H_
