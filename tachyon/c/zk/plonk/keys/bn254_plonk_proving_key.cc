#include "tachyon/c/zk/plonk/keys/bn254_plonk_proving_key.h"

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/c/zk/plonk/keys/proving_key_impl_base.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/plonk/keys/proving_key.h"

using namespace tachyon;

namespace {

// NOTE(chokobole): It assumes that proving key has univariate dense polynomial
// and evaluations.
using Poly =
    math::UnivariateDensePolynomial<math::bn254::Fr, c::math::kMaxDegree>;
using Evals = math::UnivariateEvaluations<math::bn254::Fr, c::math::kMaxDegree>;

class Bn254ProvingKeyImpl
    : public c::zk::plonk::ProvingKeyImplBase<Poly, Evals,
                                              math::bn254::G1AffinePoint> {
 public:
  explicit Bn254ProvingKeyImpl(absl::Span<const uint8_t> state)
      : c::zk::plonk::ProvingKeyImplBase<Poly, Evals,
                                         math::bn254::G1AffinePoint>(state) {}
};

using PKeyImpl = Bn254ProvingKeyImpl;

}  // namespace

tachyon_bn254_plonk_proving_key*
tachyon_bn254_plonk_proving_key_create_from_state(const uint8_t* state,
                                                  size_t state_len) {
  PKeyImpl* pkey = new PKeyImpl(absl::Span<const uint8_t>(state, state_len));
  return reinterpret_cast<tachyon_bn254_plonk_proving_key*>(pkey);
}

void tachyon_bn254_plonk_proving_key_destroy(
    tachyon_bn254_plonk_proving_key* pk) {
  delete reinterpret_cast<PKeyImpl*>(pk);
}

const tachyon_bn254_plonk_verifying_key*
tachyon_bn254_plonk_proving_key_get_verifying_key(
    const tachyon_bn254_plonk_proving_key* pk) {
  const PKeyImpl* pkey = reinterpret_cast<const PKeyImpl*>(pk);
  return reinterpret_cast<const tachyon_bn254_plonk_verifying_key*>(
      &pkey->verifying_key());
}
