#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

using namespace tachyon::math;

// NOTE(chokobole): We set |kMaxDegree| to |SIZE_MAX| on purpose to avoid
// creating variant apis corresponding to the set of each degree.
constexpr size_t kMaxDegree = SIZE_MAX;
using Evals = UnivariateEvaluations<bn254::Fr, kMaxDegree>;

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
  *reinterpret_cast<Evals&>(*evals)[i] =
      reinterpret_cast<const bn254::Fr&>(*value);
}
