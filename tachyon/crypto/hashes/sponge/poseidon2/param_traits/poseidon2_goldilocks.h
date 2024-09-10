#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_GOLDILOCKS_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_GOLDILOCKS_H_

#include <array>

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_param_traits_forward.h"
#include "tachyon/math/finite_fields/goldilocks/goldilocks.h"

namespace tachyon::crypto {

// This is taken and modified from
// https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_goldilocks.rs.
template <>
struct Poseidon2ParamsTraits<math::Goldilocks, 7, 7> {
  constexpr static std::array<math::Goldilocks, 8>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::Goldilocks{UINT64_C(0xa98811a1fed4e3a5)},
        math::Goldilocks{UINT64_C(0x1cc48b54f377e2a0)},
        math::Goldilocks{UINT64_C(0xe40cd4f6c5609a26)},
        math::Goldilocks{UINT64_C(0x11de79ebca97a4a3)},
        math::Goldilocks{UINT64_C(0x9177c73d8b7e929c)},
        math::Goldilocks{UINT64_C(0x2a6fe8085797e791)},
        math::Goldilocks{UINT64_C(0x3de6e93329f8d5ad)},
        math::Goldilocks{UINT64_C(0x3f7af9125da962fe)},
    };
  }
};

template <>
struct Poseidon2ParamsTraits<math::Goldilocks, 11, 7> {
  constexpr static std::array<math::Goldilocks, 12>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::Goldilocks{UINT64_C(0xc3b6c08e23ba9300)},
        math::Goldilocks{UINT64_C(0xd84b5de94a324fb6)},
        math::Goldilocks{UINT64_C(0x0d0c371c5b35b84f)},
        math::Goldilocks{UINT64_C(0x7964f570e7188037)},
        math::Goldilocks{UINT64_C(0x5daf18bbd996604b)},
        math::Goldilocks{UINT64_C(0x6743bc47b9595257)},
        math::Goldilocks{UINT64_C(0x5528b9362c59bb70)},
        math::Goldilocks{UINT64_C(0xac45e25b7127b68b)},
        math::Goldilocks{UINT64_C(0xa2077d7dfbb606b5)},
        math::Goldilocks{UINT64_C(0xf3faac6faee378ae)},
        math::Goldilocks{UINT64_C(0x0c6388b51545e883)},
        math::Goldilocks{UINT64_C(0xd27dbb6944917b60)},
    };
  }
};

template <>
struct Poseidon2ParamsTraits<math::Goldilocks, 15, 7> {
  constexpr static std::array<math::Goldilocks, 16>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::Goldilocks{UINT64_C(0xde9b91a467d6afc0)},
        math::Goldilocks{UINT64_C(0xc5f16b9c76a9be17)},
        math::Goldilocks{UINT64_C(0x0ab0fef2d540ac55)},
        math::Goldilocks{UINT64_C(0x3001d27009d05773)},
        math::Goldilocks{UINT64_C(0xed23b1f906d3d9eb)},
        math::Goldilocks{UINT64_C(0x5ce73743cba97054)},
        math::Goldilocks{UINT64_C(0x1c3bab944af4ba24)},
        math::Goldilocks{UINT64_C(0x2faa105854dbafae)},
        math::Goldilocks{UINT64_C(0x53ffb3ae6d421a10)},
        math::Goldilocks{UINT64_C(0xbcda9df8884ba396)},
        math::Goldilocks{UINT64_C(0xfc1273e4a31807bb)},
        math::Goldilocks{UINT64_C(0xc77952573d5142c0)},
        math::Goldilocks{UINT64_C(0x56683339a819b85e)},
        math::Goldilocks{UINT64_C(0x328fcbd8f0ddc8eb)},
        math::Goldilocks{UINT64_C(0xb5101e303fce9cb7)},
        math::Goldilocks{UINT64_C(0x774487b8c40089bb)},
    };
  }
};

template <>
struct Poseidon2ParamsTraits<math::Goldilocks, 19, 7> {
  constexpr static std::array<math::Goldilocks, 20>
  GetPoseidon2InternalDiagonalArray() {
    return {
        math::Goldilocks{UINT64_C(0x95c381fda3b1fa57)},
        math::Goldilocks{UINT64_C(0xf36fe9eb1288f42c)},
        math::Goldilocks{UINT64_C(0x89f5dcdfef277944)},
        math::Goldilocks{UINT64_C(0x106f22eadeb3e2d2)},
        math::Goldilocks{UINT64_C(0x684e31a2530e5111)},
        math::Goldilocks{UINT64_C(0x27435c5d89fd148e)},
        math::Goldilocks{UINT64_C(0x3ebed31c414dbf17)},
        math::Goldilocks{UINT64_C(0xfd45b0b2d294e3cc)},
        math::Goldilocks{UINT64_C(0x48c904473a7f6dbf)},
        math::Goldilocks{UINT64_C(0xe0d1b67809295b4d)},
        math::Goldilocks{UINT64_C(0xddd1941e9d199dcb)},
        math::Goldilocks{UINT64_C(0x8cfe534eeb742219)},
        math::Goldilocks{UINT64_C(0xa6e5261d9e3b8524)},
        math::Goldilocks{UINT64_C(0x6897ee5ed0f82c1b)},
        math::Goldilocks{UINT64_C(0x0e7dcd0739ee5f78)},
        math::Goldilocks{UINT64_C(0x493253f3d0d32363)},
        math::Goldilocks{UINT64_C(0xbb2737f5845f05c0)},
        math::Goldilocks{UINT64_C(0xa187e810b06ad903)},
        math::Goldilocks{UINT64_C(0xb635b995936c4918)},
        math::Goldilocks{UINT64_C(0x0b3694a940bd2394)},
    };
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_PARAM_TRAITS_POSEIDON2_GOLDILOCKS_H_
