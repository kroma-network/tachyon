#ifndef TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_IMPL_H_
#define TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_IMPL_H_

#include "tachyon/c/zk/plonk/halo2/bn254_halo2_ls.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl_base.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::c::zk::plonk::bn254 {

using LS = c::zk::plonk::halo2::bn254::Halo2LS;

class ProvingKeyImpl : public ProvingKeyImplBase<LS> {
 public:
  using ProvingKeyImplBase<LS>::ProvingKeyImplBase;
};

using PKeyImpl = ProvingKeyImpl;

}  // namespace tachyon::c::zk::plonk::bn254

#endif  // TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_IMPL_H_
