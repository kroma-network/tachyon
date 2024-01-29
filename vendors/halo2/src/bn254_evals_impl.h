#ifndef VENDORS_HALO2_SRC_BN254_EVALS_IMPL_H_
#define VENDORS_HALO2_SRC_BN254_EVALS_IMPL_H_

#include <utility>

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "vendors/halo2/src/degrees.h"

namespace tachyon::halo2_api::bn254 {

class EvalsImpl {
 public:
  using Evals = math::UnivariateEvaluations<math::bn254::Fr, kMaxDegree>;

  Evals& evals() { return evals_; }
  const Evals& evals() const { return evals_; }

  Evals&& TakeEvals() && { return std::move(evals_); }

 private:
  Evals evals_;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_SRC_BN254_EVALS_IMPL_H_
