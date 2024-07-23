#ifndef TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_POSEIDON2_H_
#define TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_POSEIDON2_H_

#include <array>

#include "tachyon/base/types/always_false.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::math {

template <size_t N>
std::array<BabyBear, N> GetPoseidon2BabyBearInternalDiagonalVector() {
  // TODO(chokobole): Remove this function once we can generate these parameters
  // internally.
  // This is taken and modified from
  // https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs.
  if constexpr (N == 16) {
    // Generated with rate: 15, alpha: 7, full_round: 8 and partial_round: 13.
    return {
        BabyBear{0x0a632d94}, BabyBear{0x6db657b7}, BabyBear{0x56fbdc9e},
        BabyBear{0x052b3d8a}, BabyBear{0x33745201}, BabyBear{0x5c03108c},
        BabyBear{0x0beba37b}, BabyBear{0x258c2e8b}, BabyBear{0x12029f39},
        BabyBear{0x694909ce}, BabyBear{0x6d231724}, BabyBear{0x21c3b222},
        BabyBear{0x3c0904a5}, BabyBear{0x01d6acda}, BabyBear{0x27705c83},
        BabyBear{0x5231c802},
    };
  } else if constexpr (N == 24) {
    // Generated with rate: 23, alpha: 7, full_round: 8 and partial_round: 21.
    return {
        BabyBear{0x409133f0}, BabyBear{0x1667a8a1}, BabyBear{0x06a6c7b6},
        BabyBear{0x6f53160e}, BabyBear{0x273b11d1}, BabyBear{0x03176c5d},
        BabyBear{0x72f9bbf9}, BabyBear{0x73ceba91}, BabyBear{0x5cdef81d},
        BabyBear{0x01393285}, BabyBear{0x46daee06}, BabyBear{0x065d7ba6},
        BabyBear{0x52d72d6f}, BabyBear{0x05dd05e0}, BabyBear{0x3bab4b63},
        BabyBear{0x6ada3842}, BabyBear{0x2fc5fbec}, BabyBear{0x770d61b0},
        BabyBear{0x5715aae9}, BabyBear{0x03ef0e90}, BabyBear{0x75b6c770},
        BabyBear{0x242adf5f}, BabyBear{0x00d0ca4c}, BabyBear{0x36c0e388},
    };
  } else {
    static_assert(base::AlwaysFalse<std::array<BabyBear, N>>);
  }
}

template <size_t N>
std::array<uint8_t, N> GetPoseidon2BabyBearInternalShiftVector() {
  // This is taken and modified from
  // https://github.com/Plonky3/Plonky3/blob/fde81db/baby-bear/src/poseidon2.rs.
  if constexpr (N == 15) {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
  } else if constexpr (N == 23) {
    return {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23,
    };
  } else {
    static_assert(base::AlwaysFalse<std::array<uint8_t, N>>);
  }
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_POSEIDON2_H_
