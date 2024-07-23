#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PASTA_VESTA_POSEIDON2_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PASTA_VESTA_POSEIDON2_H_

#include <array>

#include "tachyon/base/types/always_false.h"
#include "tachyon/math/elliptic_curves/pasta/vesta/fr.h"

namespace tachyon::math::vesta {

template <size_t N>
std::array<Fr, N> GetPoseidon2InternalDiagonalVector() {
  // TODO(chokobole): Remove this function once we can generate these parameters
  // internally.
  // This is taken and modified from
  // https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_vesta.rs.
  if constexpr (N == 3) {
    // Generated with rate: 2, alpha: 5, full_round: 8 and partial_round: 56.
    return {
        Fr(1),
        Fr(1),
        Fr(2),
    };
  } else {
    static_assert(base::AlwaysFalse<std::array<Fr, N>>);
  }
}

}  // namespace tachyon::math::vesta

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PASTA_VESTA_POSEIDON2_H_
