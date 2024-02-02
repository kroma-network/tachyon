#ifndef VENDORS_HALO2_INCLUDE_BN254_POLY_H_
#define VENDORS_HALO2_INCLUDE_BN254_POLY_H_

#include <utility>

#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"

namespace tachyon::halo2_api::bn254 {

class Poly {
 public:
  Poly();
  explicit Poly(tachyon_bn254_univariate_dense_polynomial* poly)
      : poly_(poly) {}
  Poly(const Poly& other) = delete;
  Poly& operator=(const Poly& other) = delete;
  ~Poly();

  tachyon_bn254_univariate_dense_polynomial* poly() { return poly_; }
  const tachyon_bn254_univariate_dense_polynomial* poly() const {
    return poly_;
  }

  tachyon_bn254_univariate_dense_polynomial* release() {
    return std::exchange(poly_, nullptr);
  }

 private:
  tachyon_bn254_univariate_dense_polynomial* poly_;
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_POLY_H_
