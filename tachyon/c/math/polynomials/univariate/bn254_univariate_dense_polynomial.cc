#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"

#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"

using namespace tachyon;

using Poly =
    math::UnivariateDensePolynomial<math::bn254::Fr, c::math::kMaxDegree>;

tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_dense_polynomial_create() {
  return c::base::c_cast(new Poly);
}

tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_dense_polynomial_clone(
    const tachyon_bn254_univariate_dense_polynomial* poly) {
  Poly* cloned_poly = new Poly(*c::base::native_cast(poly));
  return c::base::c_cast(cloned_poly);
}

void tachyon_bn254_univariate_dense_polynomial_destroy(
    tachyon_bn254_univariate_dense_polynomial* poly) {
  delete c::base::native_cast(poly);
}
