#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

using namespace tachyon::math;

// NOTE(chokobole): We set |kMaxDegree| to |SIZE_MAX| on purpose to avoid
// creating variant apis corresponding to the set of each degree.
constexpr size_t kMaxDegree = SIZE_MAX;
using Poly = UnivariateDensePolynomial<bn254::Fr, kMaxDegree>;

tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_dense_polynomial_create() {
  return reinterpret_cast<tachyon_bn254_univariate_dense_polynomial*>(new Poly);
}

tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_dense_polynomial_clone(
    const tachyon_bn254_univariate_dense_polynomial* poly) {
  Poly* cloned_poly = new Poly(*reinterpret_cast<const Poly*>(poly));
  return reinterpret_cast<tachyon_bn254_univariate_dense_polynomial*>(
      cloned_poly);
}

void tachyon_bn254_univariate_dense_polynomial_destroy(
    tachyon_bn254_univariate_dense_polynomial* poly) {
  delete reinterpret_cast<Poly*>(poly);
}
