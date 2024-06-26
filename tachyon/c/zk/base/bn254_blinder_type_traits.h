#ifndef TACHYON_C_ZK_BASE_BN254_BLINDER_TYPE_TRAITS_H_
#define TACHYON_C_ZK_BASE_BN254_BLINDER_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/base/bn254_blinder.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/zk/base/blinder.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::zk::Blinder<tachyon::math::bn254::Fr>> {
  using CType = tachyon_bn254_blinder;
};

template <>
struct TypeTraits<tachyon_bn254_blinder> {
  using NativeType = tachyon::zk::Blinder<tachyon::math::bn254::Fr>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_BASE_BN254_BLINDER_TYPE_TRAITS_H_
