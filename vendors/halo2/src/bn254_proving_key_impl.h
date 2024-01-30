#ifndef VENDORS_HALO2_SRC_BN254_PROVING_KEY_IMPL_H_
#define VENDORS_HALO2_SRC_BN254_PROVING_KEY_IMPL_H_

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "vendors/halo2/src/degrees.h"
#include "vendors/halo2/src/proving_key_impl_base.h"

namespace tachyon::halo2_api::bn254 {

class ProvingKeyImpl
    : public ProvingKeyImplBase<
          math::UnivariateDensePolynomial<math::bn254::Fr, kMaxDegree>,
          math::UnivariateEvaluations<math::bn254::Fr, kMaxDegree>,
          math::bn254::G1AffinePoint> {
 public:
  explicit ProvingKeyImpl(rust::Slice<const uint8_t> bytes)
      : ProvingKeyImplBase<
            math::UnivariateDensePolynomial<math::bn254::Fr, kMaxDegree>,
            math::UnivariateEvaluations<math::bn254::Fr, kMaxDegree>,
            math::bn254::G1AffinePoint>(bytes) {}
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_SRC_BN254_PROVING_KEY_IMPL_H_
