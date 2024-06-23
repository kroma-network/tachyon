#ifndef TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_TYPE_TRAITS_H_
#define TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<
    tachyon::zk::plonk::ConstraintSystem<tachyon::math::bn254::Fr>> {
  using CType = tachyon_bn254_plonk_constraint_system;
};

template <>
struct TypeTraits<tachyon_bn254_plonk_constraint_system> {
  using NativeType =
      tachyon::zk::plonk::ConstraintSystem<tachyon::math::bn254::Fr>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_BN254_CONSTRAINT_SYSTEM_TYPE_TRAITS_H_
