#ifndef TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_

#include <utility>
#include <vector>

#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"

namespace tachyon::zk {

template <typename Curve, size_t MaxDegree>
class ProvingKey {
 public:
  using F = typename math::AffinePoint<Curve>::ScalarField;
  using DensePoly = math::UnivariateDensePolynomial<F, MaxDegree>;
  using Evals = math::UnivariateEvaluations<F, MaxDegree>;

  ProvingKey(VerifyingKey<Curve, MaxDegree> verifying_key, DensePoly l0,
             DensePoly l_last, DensePoly l_active_row, Evals fixed_values,
             Evals fixed_polys,
             PermutationProvingKey<F, MaxDegree> permutation_proving_key)
      : verifying_key_(std::move(verifying_key)),
        l0_(std::move(l0)),
        l_last_(std::move(l_last)),
        l_active_row_(std::move(l_active_row)),
        fixed_values_(std::move(fixed_values)),
        fixed_polys_(std::move(fixed_polys)),
        permutation_proving_key_(std::move(permutation_proving_key)) {}

  const VerifyingKey<Curve, MaxDegree>& verifying_key() const {
    return verifying_key_;
  }
  const DensePoly& l0() const { return l0_; }
  const DensePoly& l_last() const { return l_last_; }
  const DensePoly& l_active_row() const { return l_active_row_; }
  const std::vector<Evals>& fixed_values() const { return fixed_values_; }
  const std::vector<Evals>& fixed_polys() const { return fixed_polys_; }
  const PermutationProvingKey<F, MaxDegree>& permutation_proving_key() const {
    return permutation_proving_key_;
  }

 private:
  VerifyingKey<Curve, MaxDegree> verifying_key_;
  DensePoly l0_;
  DensePoly l_last_;
  DensePoly l_active_row_;
  std::vector<Evals> fixed_values_;
  std::vector<Evals> fixed_polys_;
  PermutationProvingKey<F, MaxDegree> permutation_proving_key_;

  // TODO(chokobole): Evaluator ev.
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk.rs#L275
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
