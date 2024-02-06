#include "tachyon/c/math/polynomials/univariate/bn254_univariate_rational_evaluations.h"

#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

using namespace tachyon::math;

using Evals = UnivariateEvaluations<bn254::Fr, tachyon::c::math::kMaxDegree>;
using RationalEvals = UnivariateEvaluations<RationalField<bn254::Fr>,
                                            tachyon::c::math::kMaxDegree>;

tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_rational_evaluations_create() {
  return reinterpret_cast<tachyon_bn254_univariate_rational_evaluations*>(
      new RationalEvals);
}

tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_rational_evaluations_clone(
    const tachyon_bn254_univariate_rational_evaluations* evals) {
  RationalEvals* cloned_evals =
      new RationalEvals(*reinterpret_cast<const RationalEvals*>(evals));
  return reinterpret_cast<tachyon_bn254_univariate_rational_evaluations*>(
      cloned_evals);
}

void tachyon_bn254_univariate_rational_evaluations_destroy(
    tachyon_bn254_univariate_rational_evaluations* evals) {
  delete reinterpret_cast<RationalEvals*>(evals);
}

size_t tachyon_bn254_univariate_rational_evaluations_len(
    const tachyon_bn254_univariate_rational_evaluations* evals) {
  return reinterpret_cast<const RationalEvals*>(evals)->NumElements();
}

void tachyon_bn254_univariate_rational_evaluations_set_zero(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i) {
  *reinterpret_cast<RationalEvals&>(*evals)[i] =
      RationalField<bn254::Fr>::Zero();
}

void tachyon_bn254_univariate_rational_evaluations_set_trivial(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i,
    const tachyon_bn254_fr* numerator) {
  *reinterpret_cast<RationalEvals&>(*evals)[i] =
      RationalField<bn254::Fr>(reinterpret_cast<const bn254::Fr&>(*numerator));
}

void tachyon_bn254_univariate_rational_evaluations_set_rational(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t i,
    const tachyon_bn254_fr* numerator, const tachyon_bn254_fr* denominator) {
  *reinterpret_cast<RationalEvals&>(*evals)[i] = {
      reinterpret_cast<const bn254::Fr&>(*numerator),
      reinterpret_cast<const bn254::Fr&>(*denominator),
  };
}

tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_rational_evaluations_batch_evaluate(
    const tachyon_bn254_univariate_rational_evaluations* rational_evals) {
  const RationalEvals& cpp_rational_eval =
      reinterpret_cast<const RationalEvals&>(*rational_evals);
  std::vector<bn254::Fr> cpp_values(cpp_rational_eval.NumElements());
  CHECK(RationalField<bn254::Fr>::BatchEvaluate(cpp_rational_eval.evaluations(),
                                                &cpp_values));
  Evals* cpp_evals = new Evals(Evals(std::move(cpp_values)));
  return reinterpret_cast<tachyon_bn254_univariate_evaluations*>(cpp_evals);
}
