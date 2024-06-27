#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_VERIFIER_TYPE_TRAITS_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_VERIFIER_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/plonk/halo2/bn254_gwc_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_gwc_verifier.h"
#include "tachyon/c/zk/plonk/halo2/bn254_halo2_ls.h"
#include "tachyon/c/zk/plonk/halo2/verifier_impl.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<zk::plonk::halo2::VerifierImpl<
    zk::plonk::halo2::bn254::GWCPCS, zk::plonk::halo2::bn254::Halo2LS>> {
  using CType = tachyon_halo2_bn254_gwc_verifier;
};

template <>
struct TypeTraits<tachyon_halo2_bn254_gwc_verifier> {
  using NativeType =
      zk::plonk::halo2::VerifierImpl<zk::plonk::halo2::bn254::GWCPCS,
                                     zk::plonk::halo2::bn254::Halo2LS>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_VERIFIER_TYPE_TRAITS_H_
