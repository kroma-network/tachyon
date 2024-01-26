#ifndef VENDORS_HALO2_SRC_BN254_POLY_IMPL_H_
#define VENDORS_HALO2_SRC_BN254_POLY_IMPL_H_

#include <utility>

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "vendors/halo2/src/degrees.h"

namespace tachyon::halo2_api::bn254 {

class PolyImpl {
 public:
  using Poly = math::UnivariateDensePolynomial<math::bn254::Fr, kMaxDegree>;

  Poly& poly() { return poly_; }
  const Poly& poly() const { return poly_; }

  Poly&& TakePoly() && { return std::move(poly_); }

 private:
  Poly poly_;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_SRC_BN254_POLY_IMPL_H_
