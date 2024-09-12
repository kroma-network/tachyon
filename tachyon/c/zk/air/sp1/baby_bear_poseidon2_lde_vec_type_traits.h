#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_LDE_VEC_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_LDE_VEC_TYPE_TRAITS_H_

#include <vector>

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_lde_vec.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<std::vector<
    Eigen::Map<const tachyon::math::RowMajorMatrix<tachyon::math::BabyBear>>>> {
  using CType = tachyon_sp1_baby_bear_poseidon2_lde_vec;
};

template <>
struct TypeTraits<tachyon_sp1_baby_bear_poseidon2_lde_vec> {
  using NativeType =
      std::vector<Eigen::Map<const math::RowMajorMatrix<math::BabyBear>>>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_LDE_VEC_TYPE_TRAITS_H_
