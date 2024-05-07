#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_LS_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_LS_H_

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/lookup/halo2/scheme.h"

namespace tachyon::c::zk::plonk::halo2::bn254 {

using Poly = tachyon::math::UnivariateDensePolynomial<tachyon::math::bn254::Fr,
                                                      c::math::kMaxDegree>;
using Evals = tachyon::math::UnivariateEvaluations<tachyon::math::bn254::Fr,
                                                   c::math::kMaxDegree>;
using Commitment = tachyon::math::bn254::G1AffinePoint;
using LS = tachyon::zk::lookup::halo2::Scheme<Poly, Evals, Commitment>;

}  // namespace tachyon::c::zk::plonk::halo2::bn254

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_LS_H_
