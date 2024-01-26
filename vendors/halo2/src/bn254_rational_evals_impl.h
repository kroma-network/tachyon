#ifndef VENDORS_HALO2_SRC_BN254_RATIONAL_EVALS_IMPL_H_
#define VENDORS_HALO2_SRC_BN254_RATIONAL_EVALS_IMPL_H_

#include <utility>

#include "tachyon/math/base/rational_field.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "vendors/halo2/src/degrees.h"

namespace tachyon::halo2_api::bn254 {

class RationalEvalsImpl {
 public:
  using RationalEvals =
      math::UnivariateEvaluations<math::RationalField<math::bn254::Fr>,
                                  kMaxDegree>;

  RationalEvals& evals() { return evals_; }
  const RationalEvals& evals() const { return evals_; }

  RationalEvals&& TakeEvals() && { return std::move(evals_); }

 private:
  RationalEvals evals_;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_SRC_BN254_RATIONAL_EVALS_IMPL_H_
