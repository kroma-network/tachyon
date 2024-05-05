#ifndef TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_POSEIDON2_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_POSEIDON2_H_

#include <array>

#include "tachyon/base/types/always_false.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::math::bn254 {

template <size_t N>
std::array<Fr, N> GetPoseidon2InternalDiagonalVector() {
  // TODO(chokobole): remove this function once we can generate these parameters
  // internally.
  // This is taken and modified from
  // https://github.com/HorizenLabs/poseidon2/blob/bb476b9ca38198cf5092487283c8b8c5d4317c4e/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs.
  if constexpr (N == 3) {
    // Generated with rate: 2, alpha: 5, full_round: 8 and partial_round: 56.
    return {
        Fr(1),
        Fr(1),
        Fr(2),
    };
  } else {
    static_assert(base::AlwaysFalse<std::array<Fr, N>>());
  }
}

}  // namespace tachyon::math::bn254

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_BN_BN254_POSEIDON2_H_
