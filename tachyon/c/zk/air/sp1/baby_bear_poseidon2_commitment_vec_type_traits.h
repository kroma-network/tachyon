#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_TYPE_TRAITS_H_

#include <array>
#include <vector>

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_commitment_vec.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_constants.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<std::vector<std::array<
    tachyon::math::BabyBear, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>>> {
  using CType = tachyon_sp1_baby_bear_poseidon2_commitment_vec;
};

template <>
struct TypeTraits<tachyon_sp1_baby_bear_poseidon2_commitment_vec> {
  using NativeType =
      std::vector<std::array<tachyon::math::BabyBear,
                             TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_TYPE_TRAITS_H_
