#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"

using namespace tachyon;

using Evals = math::UnivariateEvaluations<math::bn254::Fr, c::math::kMaxDegree>;

tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluations_create() {
  return c::base::c_cast(new Evals);
}

tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluations_clone(
    const tachyon_bn254_univariate_evaluations* evals) {
  Evals* cloned_evals = new Evals(*c::base::native_cast(evals));
  return c::base::c_cast(cloned_evals);
}

void tachyon_bn254_univariate_evaluations_destroy(
    tachyon_bn254_univariate_evaluations* evals) {
  delete c::base::native_cast(evals);
}

size_t tachyon_bn254_univariate_evaluations_len(
    const tachyon_bn254_univariate_evaluations* evals) {
  return c::base::native_cast(evals)->NumElements();
}

void tachyon_bn254_univariate_evaluations_set_value(
    tachyon_bn254_univariate_evaluations* evals, size_t i,
    const tachyon_bn254_fr* value) {
  // NOTE(chokobole): Boundary check is the responsibility of API callers.
  c::base::native_cast(*evals).at(i) = c::base::native_cast(*value);
}
