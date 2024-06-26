#ifndef TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_TYPE_TRAITS_H_
#define TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::zk::plonk::VerifyingKey<
    tachyon::math::bn254::Fr, tachyon::math::bn254::G1AffinePoint>> {
  using CType = tachyon_bn254_plonk_verifying_key;
};

template <>
struct TypeTraits<tachyon_bn254_plonk_verifying_key> {
  using NativeType =
      tachyon::zk::plonk::VerifyingKey<tachyon::math::bn254::Fr,
                                       tachyon::math::bn254::G1AffinePoint>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_VERIFYING_KEY_TYPE_TRAITS_H_
