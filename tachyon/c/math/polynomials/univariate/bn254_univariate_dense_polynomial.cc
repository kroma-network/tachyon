#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

using namespace tachyon::math;

using Poly = UnivariateDensePolynomial<bn254::Fr, tachyon::c::math::kMaxDegree>;

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
