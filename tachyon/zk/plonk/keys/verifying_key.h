#ifndef TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_

#include <stddef.h>

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/plonk/commitment.h"
#include "tachyon/zk/plonk/constraint_system.h"
#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

namespace tachyon::zk {

template <typename Curve, size_t MaxDegree>
class VerifyingKey {
 public:
  using F = typename Curve::ScalarField;

  VerifyingKey(
      std::unique_ptr<math::UnivariateEvaluationDomain<F, MaxDegree>> domain,
      Commitments<Curve> fixed_commitments,
      PermutationVerifyingKey<Curve> permutation_verifying_key,
      ConstraintSystem<F> constraint_system)
      : domain_(std::move(domain)),
        fixed_commitments_(std::move(fixed_commitments)),
        permutation_verifying_Key_(std::move(permutation_verifying_key)),
        constraint_system_(std::move(constraint_system)) {}

  const math::UnivariateEvaluationDomain<F, MaxDegree>* domain() const {
    return domain_.get();
  }

  const Commitments<Curve>& fixed_commitments() const {
    return fixed_commitments_;
  }

  const PermutationVerifyingKey<Curve>& permutation_verifying_key() const {
    return permutation_verifying_Key_;
  }

  const ConstraintSystem<F>& constraint_system() const {
    return constraint_system_;
  }

  const F& transcript_repr() const { return transcript_repr_; }

 private:
  std::unique_ptr<math::UnivariateEvaluationDomain<F, MaxDegree>> domain_;
  Commitments<Curve> fixed_commitments_;
  PermutationVerifyingKey<Curve> permutation_verifying_Key_;
  ConstraintSystem<F> constraint_system_;
  // The representative of this |VerifyingKey| in transcripts.
  F transcript_repr_ = F::Zero();
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_
