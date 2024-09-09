#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_MERSENNE31_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_MERSENNE31_H_

#include <array>

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_param_traits_forward.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"

namespace tachyon::crypto {

// This is taken and modified from
// https://github.com/Plonky3/Plonky3/blob/fde81db/mersenne-31/src/poseidon2.rs.
template <>
struct Poseidon2ParamsTraits<math::Mersenne31, 15, 7> {
  constexpr static std::array<uint8_t, 15> GetPoseidon2InternalShiftArray() {
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16};
  }
};

template <>
struct Poseidon2ParamsTraits<math::Mersenne31, 23, 7> {
  constexpr static std::array<uint8_t, 23> GetPoseidon2InternalShiftArray() {
    return {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    };
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_MERSENNE31_H_
