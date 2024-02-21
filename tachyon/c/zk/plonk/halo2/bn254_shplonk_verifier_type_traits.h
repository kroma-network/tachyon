#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_TYPE_TRAITS_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_verifier.h"
#include "tachyon/c/zk/plonk/halo2/verifier_impl.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<
    zk::plonk::halo2::VerifierImpl<zk::plonk::halo2::bn254::SHPlonkPCS>> {
  using CType = tachyon_halo2_bn254_shplonk_verifier;
};

template <>
struct TypeTraits<tachyon_halo2_bn254_shplonk_verifier> {
  using NativeType =
      zk::plonk::halo2::VerifierImpl<zk::plonk::halo2::bn254::SHPlonkPCS>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_TYPE_TRAITS_H_
