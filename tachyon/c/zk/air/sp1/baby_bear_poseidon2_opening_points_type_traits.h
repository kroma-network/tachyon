#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENING_POINTS_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENING_POINTS_TYPE_TRAITS_H_

#include <vector>

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opening_points.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<
    std::vector<std::vector<std::vector<tachyon::math::BabyBear4>>>> {
  using CType = tachyon_sp1_baby_bear_poseidon2_opening_points;
};

template <>
struct TypeTraits<tachyon_sp1_baby_bear_poseidon2_opening_points> {
  using NativeType =
      std::vector<std::vector<std::vector<tachyon::math::BabyBear4>>>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_OPENING_POINTS_TYPE_TRAITS_H_
