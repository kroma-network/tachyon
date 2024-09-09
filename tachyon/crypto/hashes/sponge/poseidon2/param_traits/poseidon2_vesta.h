#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_VESTA_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_VESTA_H_

#include <array>

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_param_traits_forward.h"
#include "tachyon/math/elliptic_curves/pasta/vesta/fr.h"

namespace tachyon::crypto {

// This is taken and modified from
// https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_vesta.rs.
template <>
struct Poseidon2ParamsTraits<math::vesta::Fr, 2, 5> {
  constexpr static std::array<math::vesta::Fr, 3>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::vesta::Fr(1),
        math::vesta::Fr(1),
        math::vesta::Fr(2),
    };
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_VESTA_H_
