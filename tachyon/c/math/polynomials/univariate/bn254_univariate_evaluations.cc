#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

using namespace tachyon::math;

using Evals = UnivariateEvaluations<bn254::Fr, tachyon::c::math::kMaxDegree>;

tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluations_create() {
  return reinterpret_cast<tachyon_bn254_univariate_evaluations*>(new Evals);
}

tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluations_clone(
    const tachyon_bn254_univariate_evaluations* evals) {
  Evals* cloned_evals = new Evals(*reinterpret_cast<const Evals*>(evals));
  return reinterpret_cast<tachyon_bn254_univariate_evaluations*>(cloned_evals);
}

void tachyon_bn254_univariate_evaluations_destroy(
    tachyon_bn254_univariate_evaluations* evals) {
  delete reinterpret_cast<Evals*>(evals);
}

size_t tachyon_bn254_univariate_evaluations_len(
    const tachyon_bn254_univariate_evaluations* evals) {
  return reinterpret_cast<const Evals*>(evals)->NumElements();
}

void tachyon_bn254_univariate_evaluations_set_value(
    tachyon_bn254_univariate_evaluations* evals, size_t i,
    const tachyon_bn254_fr* value) {
  // NOTE(chokobole): Boundary check is the responsibility of API callers.
  reinterpret_cast<Evals&>(*evals).at(i) =
      tachyon::c::base::native_cast(*value);
}
