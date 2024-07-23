#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PASTA_PALLAS_POSEIDON2_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PASTA_PALLAS_POSEIDON2_H_

#include <array>

#include "tachyon/base/types/always_false.h"
#include "tachyon/math/elliptic_curves/pasta/pallas/fr.h"

namespace tachyon::math::pallas {

template <size_t N>
std::array<Fr, N> GetPoseidon2InternalDiagonalVector() {
  // TODO(chokobole): Remove this function once we can generate these parameters
  // internally.
  // This is taken and modified from
  // https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_pallas.rs.
  if constexpr (N == 3) {
    // Generated with rate: 2, alpha: 5, full_round: 8 and partial_round: 56.
    return {
        Fr(1),
        Fr(1),
        Fr(2),
    };
  } else if constexpr (N == 4) {
    // Generated with rate: 3, alpha: 5, full_round: 8 and partial_round: 56.
    return {
        // clang-format off
        Fr::FromHexString("0x0767b051e5b6358fd12f217aae53bb9dac9a72a9f6a16fdde8f36e715bb27f51"),
        Fr::FromHexString("0x2a59f16a37626bdd5536c5546f046b608c777734990103996730611728cfef21"),
        Fr::FromHexString("0x2388405f3a1e87a1fd3183bb12a89c71b37555b4db6a4306e1f05322217ee15c"),
        Fr::FromHexString("0x0e7c7e19ad92352c35e4d302828f64de68750dac64cbd944f0eba6c0ed003757"),
        // clang-format on
    };
  } else if constexpr (N == 8) {
    // Generated with rate: 7, alpha: 5, full_round: 8 and partial_round: 57.
    return {
        // clang-format off
        Fr::FromHexString("0x2527e8a83e49ae6bf3c8e459d5220e34d84aa49ce14f2dc401273cebdec65067"),
        Fr::FromHexString("0x0e4a24b206b7494d2437d3e0fd1deeae8a943ccd836e0f959aaeccebb3068859"),
        Fr::FromHexString("0x03c9638e9b8ad067e7033ed3aef5e185fcfa3959f82283bbf00c1bf0ea40fe45"),
        Fr::FromHexString("0x09f7633edc22a16de93a8676260b507aa44aa8c57565bc5d21543897be56c100"),
        Fr::FromHexString("0x06dfc4a91b7acb8ef203a8bc6b850290b1e272a594512ac0d1c9d3a56c8a7921"),
        Fr::FromHexString("0x020e0af80c2a8e2aab6dcf5d8e94e71d24156e3123a16fdf8fd80471776e3551"),
        Fr::FromHexString("0x24dfd0278f203a55322e94b290ae4269bbe76aa5531921f5f87a8d2d736dbb9c"),
        Fr::FromHexString("0x08aa91c42dea2206ff4e601a1f49c009d18acc891ccb78529856db2a49664b2d"),
        // clang-format on
    };
  } else {
    static_assert(base::AlwaysFalse<std::array<Fr, N>>);
  }
}

}  // namespace tachyon::math::pallas

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PASTA_PALLAS_POSEIDON2_H_
