#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_IMPL_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_IMPL_H_

#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_pcs.h"
#include "tachyon/c/zk/plonk/halo2/verifier_impl_base.h"

namespace tachyon::c::zk::plonk::halo2::bn254 {

class SHPlonkVerifierImpl : public VerifierImplBase<PCS> {
 public:
  using VerifierImplBase<PCS>::VerifierImplBase;
};

}  // namespace tachyon::c::zk::plonk::halo2::bn254

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_VERIFIER_IMPL_H_
