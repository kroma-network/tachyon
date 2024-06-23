#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_TYPE_TRAITS_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_TYPE_TRAITS_H_

#include <vector>

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<
    std::vector<std::vector<std::vector<tachyon::math::bn254::Fr>>>> {
  using CType = tachyon_halo2_bn254_instance_columns_vec;
};

template <>
struct TypeTraits<tachyon_halo2_bn254_instance_columns_vec> {
  using NativeType =
      std::vector<std::vector<std::vector<tachyon::math::bn254::Fr>>>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_INSTANCE_COLUMNS_VEC_TYPE_TRAITS_H_
