#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_BABY_BEAR_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_BABY_BEAR_H_

#include <array>

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_param_traits_forward.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::crypto {

// This is taken and modified from
// https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs.
// and
// https://github.com/Plonky3/Plonky3/blob/fde81db/baby-bear/src/poseidon2.rs.
template <>
struct Poseidon2ParamsTraits<math::BabyBear, 15, 7> {
  constexpr static std::array<math::BabyBear, 16>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::BabyBear{0x0a632d94}, math::BabyBear{0x6db657b7},
        math::BabyBear{0x56fbdc9e}, math::BabyBear{0x052b3d8a},
        math::BabyBear{0x33745201}, math::BabyBear{0x5c03108c},
        math::BabyBear{0x0beba37b}, math::BabyBear{0x258c2e8b},
        math::BabyBear{0x12029f39}, math::BabyBear{0x694909ce},
        math::BabyBear{0x6d231724}, math::BabyBear{0x21c3b222},
        math::BabyBear{0x3c0904a5}, math::BabyBear{0x01d6acda},
        math::BabyBear{0x27705c83}, math::BabyBear{0x5231c802},
    };
  }
  constexpr static std::array<uint8_t, 15> GetPoseidon2InternalShiftArray() {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
  }
};

template <>
struct Poseidon2ParamsTraits<math::BabyBear, 23, 7> {
  constexpr static std::array<math::BabyBear, 24>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::BabyBear{0x0a632d94}, math::BabyBear{0x6db657b7},
        math::BabyBear{0x56fbdc9e}, math::BabyBear{0x052b3d8a},
        math::BabyBear{0x33745201}, math::BabyBear{0x5c03108c},
        math::BabyBear{0x0beba37b}, math::BabyBear{0x258c2e8b},
        math::BabyBear{0x12029f39}, math::BabyBear{0x694909ce},
        math::BabyBear{0x6d231724}, math::BabyBear{0x21c3b222},
        math::BabyBear{0x3c0904a5}, math::BabyBear{0x01d6acda},
        math::BabyBear{0x27705c83}, math::BabyBear{0x5231c802},
    };
  }
  constexpr static std::array<uint8_t, 23> GetPoseidon2InternalShiftArray() {
    return {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23,
    };
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_BABY_BEAR_H_
