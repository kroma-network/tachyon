#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"

#include <utility>

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

using namespace tachyon::math;

using Domain =
    UnivariateEvaluationDomain<bn254::Fr, tachyon::c::math::kMaxDegree>;
using RationalEvals = UnivariateEvaluations<RationalField<bn254::Fr>,
                                            tachyon::c::math::kMaxDegree>;

tachyon_bn254_univariate_evaluation_domain*
tachyon_bn254_univariate_evaluation_domain_create(size_t num_coeffs) {
  return reinterpret_cast<tachyon_bn254_univariate_evaluation_domain*>(
      Domain::Create(num_coeffs).release());
}

void tachyon_bn254_univariate_evaluation_domain_destroy(
    tachyon_bn254_univariate_evaluation_domain* domain) {
  delete reinterpret_cast<Domain*>(domain);
}

tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluation_domain_empty_evals(
    const tachyon_bn254_univariate_evaluation_domain* domain) {
  return reinterpret_cast<tachyon_bn254_univariate_evaluations*>(
      new Domain::Evals(
          reinterpret_cast<const Domain*>(domain)->Zero<Domain::Evals>()));
}

tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_evaluation_domain_empty_poly(
    const tachyon_bn254_univariate_evaluation_domain* domain) {
  return reinterpret_cast<tachyon_bn254_univariate_dense_polynomial*>(
      new Domain::DensePoly(
          reinterpret_cast<const Domain*>(domain)->Zero<Domain::DensePoly>()));
}

tachyon_bn254_univariate_rational_evaluations*
tachyon_bn254_univariate_evaluation_domain_empty_rational_evals(
    const tachyon_bn254_univariate_evaluation_domain* domain) {
  return reinterpret_cast<tachyon_bn254_univariate_rational_evaluations*>(
      new RationalEvals(
          reinterpret_cast<const Domain*>(domain)->Zero<RationalEvals>()));
}

tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluation_domain_fft(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    const tachyon_bn254_univariate_dense_polynomial* poly) {
  Domain::Evals* evals =
      new Domain::Evals(reinterpret_cast<const Domain*>(domain)->FFT(
          reinterpret_cast<const Domain::DensePoly&>(*poly)));
  return reinterpret_cast<tachyon_bn254_univariate_evaluations*>(evals);
}

tachyon_bn254_univariate_evaluations*
tachyon_bn254_univariate_evaluation_domain_fft_inplace(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    tachyon_bn254_univariate_dense_polynomial* poly) {
  Domain::Evals* evals =
      new Domain::Evals(reinterpret_cast<const Domain*>(domain)->FFT(
          reinterpret_cast<Domain::DensePoly&&>(std::move(*poly))));
  return reinterpret_cast<tachyon_bn254_univariate_evaluations*>(evals);
}

tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_evaluation_domain_ifft(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    const tachyon_bn254_univariate_evaluations* evals) {
  Domain::DensePoly* poly =
      new Domain::DensePoly(reinterpret_cast<const Domain*>(domain)->IFFT(
          reinterpret_cast<const Domain::Evals&>(*evals)));
  return reinterpret_cast<tachyon_bn254_univariate_dense_polynomial*>(poly);
}

tachyon_bn254_univariate_dense_polynomial*
tachyon_bn254_univariate_evaluation_domain_ifft_inplace(
    const tachyon_bn254_univariate_evaluation_domain* domain,
    tachyon_bn254_univariate_evaluations* evals) {
  Domain::DensePoly* poly =
      new Domain::DensePoly(reinterpret_cast<const Domain*>(domain)->IFFT(
          reinterpret_cast<Domain::Evals&&>(std::move(*evals))));
  return reinterpret_cast<tachyon_bn254_univariate_dense_polynomial*>(poly);
}
