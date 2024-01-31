#include "vendors/halo2/include/bn254_poly.h"

namespace tachyon::halo2_api::bn254 {

Poly::Poly() : poly_(tachyon_bn254_univariate_dense_polynomial_create()) {}

Poly::~Poly() { tachyon_bn254_univariate_dense_polynomial_destroy(poly_); }

}  // namespace tachyon::halo2_api::bn254
