#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFIER_DATA_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFIER_DATA_H_

#include "tachyon/zk/plonk/base/multi_phase_evaluations.h"

namespace tachyon::zk::plonk {

template <typename F, typename C>
struct VanishingVerifierData : public MultiPhaseEvaluations<F> {
  VanishingVerifierData(absl::Span<const C> fixed_commitments,
                        absl::Span<const C> advice_commitments,
                        absl::Span<const C> instance_commitments,
                        absl::Span<const F> fixed_evals,
                        absl::Span<const F> advice_evals,
                        absl::Span<const F> instance_evals,
                        absl::Span<const F> challenges,
                        absl::Span<const C> h_poly_commitments,
                        const C& random_poly_commitment, const F& random_eval)
      : MultiPhaseEvaluations<F>(fixed_evals, advice_evals, instance_evals,
                                 challenges),
        fixed_commitments(fixed_commitments),
        advice_commitments(advice_commitments),
        instance_commitments(instance_commitments),
        h_poly_commitments(h_poly_commitments),
        random_poly_commitment(random_poly_commitment),
        random_eval(random_eval) {}

  absl::Span<const C> fixed_commitments;
  absl::Span<const C> advice_commitments;
  absl::Span<const C> instance_commitments;
  absl::Span<const C> h_poly_commitments;
  const C& random_poly_commitment;
  const F& random_eval;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFIER_DATA_H_
