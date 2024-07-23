#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_POSEIDON2_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_POSEIDON2_H_

#include <array>

#include "tachyon/base/types/always_false.h"
#include "tachyon/math/finite_fields/goldilocks/goldilocks.h"

namespace tachyon::math {

template <size_t N>
std::array<Goldilocks, N> GetPoseidon2GoldilocksInternalDiagonalVector() {
  // TODO(chokobole): Remove this function once we can generate these parameters
  // internally.
  // This is taken and modified from
  // https://github.com/HorizenLabs/poseidon2/blob/bb476b9/plain_implementations/src/poseidon2/poseidon2_instance_goldilocks.rs.
  if constexpr (N == 8) {
    // Generated with rate: 7, alpha: 7, full_round: 8 and partial_round: 22.
    return {
        Goldilocks{UINT64_C(0xa98811a1fed4e3a5)},
        Goldilocks{UINT64_C(0x1cc48b54f377e2a0)},
        Goldilocks{UINT64_C(0xe40cd4f6c5609a26)},
        Goldilocks{UINT64_C(0x11de79ebca97a4a3)},
        Goldilocks{UINT64_C(0x9177c73d8b7e929c)},
        Goldilocks{UINT64_C(0x2a6fe8085797e791)},
        Goldilocks{UINT64_C(0x3de6e93329f8d5ad)},
        Goldilocks{UINT64_C(0x3f7af9125da962fe)},
    };
  } else if constexpr (N == 12) {
    // Generated with rate: 11, alpha: 7, full_round: 8 and partial_round: 22.
    return {
        Goldilocks{UINT64_C(0xc3b6c08e23ba9300)},
        Goldilocks{UINT64_C(0xd84b5de94a324fb6)},
        Goldilocks{UINT64_C(0x0d0c371c5b35b84f)},
        Goldilocks{UINT64_C(0x7964f570e7188037)},
        Goldilocks{UINT64_C(0x5daf18bbd996604b)},
        Goldilocks{UINT64_C(0x6743bc47b9595257)},
        Goldilocks{UINT64_C(0x5528b9362c59bb70)},
        Goldilocks{UINT64_C(0xac45e25b7127b68b)},
        Goldilocks{UINT64_C(0xa2077d7dfbb606b5)},
        Goldilocks{UINT64_C(0xf3faac6faee378ae)},
        Goldilocks{UINT64_C(0x0c6388b51545e883)},
        Goldilocks{UINT64_C(0xd27dbb6944917b60)},
    };
  } else if constexpr (N == 16) {
    // Generated with rate: 15, alpha: 7, full_round: 8 and partial_round: 22.
    return {
        Goldilocks{UINT64_C(0xde9b91a467d6afc0)},
        Goldilocks{UINT64_C(0xc5f16b9c76a9be17)},
        Goldilocks{UINT64_C(0x0ab0fef2d540ac55)},
        Goldilocks{UINT64_C(0x3001d27009d05773)},
        Goldilocks{UINT64_C(0xed23b1f906d3d9eb)},
        Goldilocks{UINT64_C(0x5ce73743cba97054)},
        Goldilocks{UINT64_C(0x1c3bab944af4ba24)},
        Goldilocks{UINT64_C(0x2faa105854dbafae)},
        Goldilocks{UINT64_C(0x53ffb3ae6d421a10)},
        Goldilocks{UINT64_C(0xbcda9df8884ba396)},
        Goldilocks{UINT64_C(0xfc1273e4a31807bb)},
        Goldilocks{UINT64_C(0xc77952573d5142c0)},
        Goldilocks{UINT64_C(0x56683339a819b85e)},
        Goldilocks{UINT64_C(0x328fcbd8f0ddc8eb)},
        Goldilocks{UINT64_C(0xb5101e303fce9cb7)},
        Goldilocks{UINT64_C(0x774487b8c40089bb)},
    };
  } else if constexpr (N == 20) {
    // Generated with rate: 19, alpha: 7, full_round: 8 and partial_round: 22.
    return {
        Goldilocks{UINT64_C(0x95c381fda3b1fa57)},
        Goldilocks{UINT64_C(0xf36fe9eb1288f42c)},
        Goldilocks{UINT64_C(0x89f5dcdfef277944)},
        Goldilocks{UINT64_C(0x106f22eadeb3e2d2)},
        Goldilocks{UINT64_C(0x684e31a2530e5111)},
        Goldilocks{UINT64_C(0x27435c5d89fd148e)},
        Goldilocks{UINT64_C(0x3ebed31c414dbf17)},
        Goldilocks{UINT64_C(0xfd45b0b2d294e3cc)},
        Goldilocks{UINT64_C(0x48c904473a7f6dbf)},
        Goldilocks{UINT64_C(0xe0d1b67809295b4d)},
        Goldilocks{UINT64_C(0xddd1941e9d199dcb)},
        Goldilocks{UINT64_C(0x8cfe534eeb742219)},
        Goldilocks{UINT64_C(0xa6e5261d9e3b8524)},
        Goldilocks{UINT64_C(0x6897ee5ed0f82c1b)},
        Goldilocks{UINT64_C(0x0e7dcd0739ee5f78)},
        Goldilocks{UINT64_C(0x493253f3d0d32363)},
        Goldilocks{UINT64_C(0xbb2737f5845f05c0)},
        Goldilocks{UINT64_C(0xa187e810b06ad903)},
        Goldilocks{UINT64_C(0xb635b995936c4918)},
        Goldilocks{UINT64_C(0x0b3694a940bd2394)},
        Goldilocks{UINT64_C(0x56683339a819b85e)},
        Goldilocks{UINT64_C(0x328fcbd8f0ddc8eb)},
        Goldilocks{UINT64_C(0xb5101e303fce9cb7)},
        Goldilocks{UINT64_C(0x774487b8c40089bb)},
    };
  } else {
    static_assert(base::AlwaysFalse<std::array<Goldilocks, N>>);
  }
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_POSEIDON2_H_
