#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_PALLAS_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_PALLAS_H_

#include <array>

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_param_traits_forward.h"
#include "tachyon/math/elliptic_curves/pasta/pallas/fr.h"

namespace tachyon::crypto {

// This is taken and modified from
// https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_pallas.rs.
template <>
struct Poseidon2ParamsTraits<math::Pallas::Fr, 2, 5> {
  constexpr static std::array<math::Pallas::Fr, 3>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::Pallas::Fr(1),
        math::Pallas::Fr(1),
        math::Pallas::Fr(2),
    };
  }
};

template <>
struct Poseidon2ParamsTraits<math::Pallas::Fr, 3, 5> {
  constexpr static std::array<math::Pallas::Fr, 4>
  GetPoseidon2InternalDiagonalArray() {
    return {
        // clang-format off
        math::Pallas::Fr::FromHexString("0x0767b051e5b6358fd12f217aae53bb9dac9a72a9f6a16fdde8f36e715bb27f51"),
        math::Pallas::Fr::FromHexString("0x2a59f16a37626bdd5536c5546f046b608c777734990103996730611728cfef21"),
        math::Pallas::Fr::FromHexString("0x2388405f3a1e87a1fd3183bb12a89c71b37555b4db6a4306e1f05322217ee15c"),
        math::Pallas::Fr::FromHexString("0x0e7c7e19ad92352c35e4d302828f64de68750dac64cbd944f0eba6c0ed003757"),
        // clang-format on
    };
  }
};

template <>
struct Poseidon2ParamsTraits<math::Pallas::Fr, 7, 5> {
  constexpr static std::array<math::Pallas::Fr, 8>
  GetPoseidon2InternalDiagonalArray() {
    return {
        // clang-format off
        math::Pallas::Fr::FromHexString("0x2527e8a83e49ae6bf3c8e459d5220e34d84aa49ce14f2dc401273cebdec65067"),
        math::Pallas::Fr::FromHexString("0x0e4a24b206b7494d2437d3e0fd1deeae8a943ccd836e0f959aaeccebb3068859"),
        math::Pallas::Fr::FromHexString("0x03c9638e9b8ad067e7033ed3aef5e185fcfa3959f82283bbf00c1bf0ea40fe45"),
        math::Pallas::Fr::FromHexString("0x09f7633edc22a16de93a8676260b507aa44aa8c57565bc5d21543897be56c100"),
        math::Pallas::Fr::FromHexString("0x06dfc4a91b7acb8ef203a8bc6b850290b1e272a594512ac0d1c9d3a56c8a7921"),
        math::Pallas::Fr::FromHexString("0x020e0af80c2a8e2aab6dcf5d8e94e71d24156e3123a16fdf8fd80471776e3551"),
        math::Pallas::Fr::FromHexString("0x24dfd0278f203a55322e94b290ae4269bbe76aa5531921f5f87a8d2d736dbb9c"),
        math::Pallas::Fr::FromHexString("0x08aa91c42dea2206ff4e601a1f49c009d18acc891ccb78529856db2a49664b2d"),
        // clang-format on
    };
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_PALLAS_H_
