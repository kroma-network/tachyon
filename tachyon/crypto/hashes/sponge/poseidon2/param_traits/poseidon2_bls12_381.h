#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_BLS12_381_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_BLS12_381_H_

#include <array>

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_param_traits_forward.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_318/fr.h"

namespace tachyon::crypto {

// This is taken and modified from
// https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_bls12.rs.
template <>
struct Poseidon2ParamsTraits<math::bls12_381::Fr, 1, 5> {
  constexpr static std::array<math::bls12_381::Fr, 2>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::bls12_381::Fr(1),
        math::bls12_381::Fr(2),
    };
  }
};

template <>
struct Poseidon2ParamsTraits<math::bls12_381::Fr, 2, 5> {
  constexpr static std::array<math::bls12_381::Fr, 3>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::bls12_381::Fr(1),
        math::bls12_381::Fr(1),
        math::bls12_381::Fr(2),
    };
  }
};

template <>
struct Poseidon2ParamsTraits<math::bls12_381::Fr, 3, 5> {
  constexpr static std::array<math::bls12_381::Fr, 4>
  GetPoseidon2InternalDiagonalArray() {
    return {
        // clang-format off
        math::bls12_381::Fr::FromHexString("0x07564ad691bf01c8601d68757a561d224f00f313ada673ab83e6255fb4fd5b3d"),
        math::bls12_381::Fr::FromHexString("0x6184e3be38549f7c0850cd069b32f6decbfde312dd4b8c18349b1b3776a6eaa4"),
        math::bls12_381::Fr::FromHexString("0x419289088178ad742be6f78425c0156b6546a18fd338f0169937dea46cfb64d2"),
        math::bls12_381::Fr::FromHexString("0x3244cdec173b71a4659e2529b499362dac10cb2fd17562860c8bb9d0fd45b787"),
        // clang-format on
    };
  }
};

template <>
struct Poseidon2ParamsTraits<math::bls12_381::Fr, 4, 5> {
  constexpr static std::array<math::bls12_381::Fr, 5>
  GetPoseidon2InternalDiagonalArray() {
    return {
        // clang-format off
        math::bls12_381::Fr::FromHexString("0x1118b610c2544efa26b70d9d60ca6ca362afcfff12436cf3b0f8a3ec5895d9ea"),
        math::bls12_381::Fr::FromHexString("0x5ba288c5197e71745a8fde16aca575e379dcc19f21042d8b9375e478f809325b"),
        math::bls12_381::Fr::FromHexString("0x079a987d87d7c80d5f4a3b4018517c50f5067ecb516f6bd14d79eabaa8349e62"),
        math::bls12_381::Fr::FromHexString("0x4c6497b0b99e1f1af4ec0322dc38869b2dfb79db3ab5fa68936cc8b6025aad1f"),
        math::bls12_381::Fr::FromHexString("0x483b5c5071e90c98bd353556453f04113442f29a1c4c236b4ca31890136bee4d"),
        math::bls12_381::Fr::FromHexString("0x3ef76c8bae0aa755dde594d8ec22b157f913323e5b29bbd0652e4b74973ac8f9"),
        math::bls12_381::Fr::FromHexString("0x091767b280c59a58a39f293bfc22ae944cb921c2efa240262b5b66312724f20b"),
        math::bls12_381::Fr::FromHexString("0x45ef82a5684137e5fc9613e0581cb65b5ad3d43470eacf0f060e1711c4c57623"),
        // clang-format on
    };
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_BLS12_381_H_
