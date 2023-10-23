#ifndef TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_

#include <vector>

#include "tachyon/math/polynomials/polynomial.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

namespace tachyon::zk {

template <typename Curve, size_t MaxDegree>
struct ProvingKey {
  using ScalarField = typename math::AffinePoint<Curve>::ScalarField;
  using Poly = math::UnivariateDensePolynomial<ScalarField, MaxDegree>;
  using Coeffs = math::UnivariateDenseCoefficients<ScalarField, MaxDegree>;

  VerifyingKey<Curve, MaxDegree> vk;
  Poly l0;
  Poly l_last;
  Poly l_active_row;
  std::vector<Coeffs> fixed_values;
  std::vector<Poly> fixed_polys;
  // TODO(chokobole): Add permutation and ev.
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk.rs#L274-L275
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
