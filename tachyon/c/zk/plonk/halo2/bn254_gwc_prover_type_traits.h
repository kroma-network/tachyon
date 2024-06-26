#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_PROVER_TYPE_TRAITS_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_PROVER_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/plonk/halo2/bn254_gwc_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_gwc_prover.h"
#include "tachyon/c/zk/plonk/halo2/bn254_ls.h"
#include "tachyon/c/zk/plonk/halo2/kzg_family_prover_impl.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<zk::plonk::halo2::KZGFamilyProverImpl<
    zk::plonk::halo2::bn254::GWCPCS, zk::plonk::halo2::bn254::LS>> {
  using CType = tachyon_halo2_bn254_gwc_prover;
};

template <>
struct TypeTraits<tachyon_halo2_bn254_gwc_prover> {
  using NativeType =
      zk::plonk::halo2::KZGFamilyProverImpl<zk::plonk::halo2::bn254::GWCPCS,
                                            zk::plonk::halo2::bn254::LS>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_GWC_PROVER_TYPE_TRAITS_H_
