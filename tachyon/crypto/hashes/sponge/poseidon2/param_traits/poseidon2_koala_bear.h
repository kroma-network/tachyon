#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_KOALA_BEAR_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_KOALA_BEAR_H_

#include <array>

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_param_traits_forward.h"
#include "tachyon/math/finite_fields/koala_bear/koala_bear.h"

namespace tachyon::crypto {

// This is taken and modified from
// https://github.com/Plonky3/Plonky3/blob/fde81db/koala-bear/src/poseidon2.rs.
template <>
struct Poseidon2ParamsTraits<math::KoalaBear, 15, 7> {
  constexpr static std::array<uint8_t, 15> GetPoseidon2InternalShiftArray() {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
  }
};

template <>
struct Poseidon2ParamsTraits<math::KoalaBear, 23, 7> {
  constexpr static std::array<uint8_t, 23> GetPoseidon2InternalShiftArray() {
    return {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23,
    };
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_KOALA_BEAR_H_
