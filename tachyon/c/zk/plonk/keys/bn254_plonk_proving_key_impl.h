#ifndef TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_IMPL_H_
#define TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_IMPL_H_

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl_base.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::c::zk::plonk::bn254 {

// NOTE(chokobole): It assumes that proving key has univariate dense polynomial
// and evaluations.
using Poly = tachyon::math::UnivariateDensePolynomial<tachyon::math::bn254::Fr,
                                                      c::math::kMaxDegree>;
using Evals = tachyon::math::UnivariateEvaluations<tachyon::math::bn254::Fr,
                                                   c::math::kMaxDegree>;

class ProvingKeyImpl
    : public ProvingKeyImplBase<Poly, Evals,
                                tachyon::math::bn254::G1AffinePoint> {
 public:
  using ProvingKeyImplBase<
      Poly, Evals, tachyon::math::bn254::G1AffinePoint>::ProvingKeyImplBase;
};

using PKeyImpl = ProvingKeyImpl;

}  // namespace tachyon::c::zk::plonk::bn254

#endif  // TACHYON_C_ZK_PLONK_KEYS_BN254_PLONK_PROVING_KEY_IMPL_H_
