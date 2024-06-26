#ifndef TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_TYPE_TRAITS_H_
#define TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key_impl.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<zk::plonk::bn254::ProvingKeyImpl> {
  using CType = tachyon_bn254_plonk_proving_key;
};

template <>
struct TypeTraits<tachyon_bn254_plonk_proving_key> {
  using NativeType = zk::plonk::bn254::ProvingKeyImpl;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_TYPE_TRAITS_H_
