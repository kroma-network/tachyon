#ifndef TACHYON_ZK_SHUFFLE_VERIFIER_DATA_H_
#define TACHYON_ZK_SHUFFLE_VERIFIER_DATA_H_

#include "tachyon/zk/plonk/base/multi_phase_evaluations.h"

namespace tachyon::zk::shuffle {

template <typename F, typename C>
struct VerifierData : public plonk::MultiPhaseEvaluations<F> {
  VerifierData(absl::Span<const F> fixed_evals,
               absl::Span<const F> advice_evals,
               absl::Span<const F> instance_evals,
               absl::Span<const F> challenges,
               absl::Span<const C> grand_product_commitments,
               absl::Span<const F> grand_product_evals,
               absl::Span<const F> grand_product_next_evals, const F& theta,
               const F& gamma)
      : plonk::MultiPhaseEvaluations<F>(fixed_evals, advice_evals,
                                        instance_evals, challenges),
        grand_product_commitments(grand_product_commitments),
        grand_product_evals(grand_product_evals),
        grand_product_next_evals(grand_product_next_evals),
        theta(theta),
        gamma(gamma) {}

  // [Zₛ,ᵢ(τ)]₁
  absl::Span<const C> grand_product_commitments;
  // Zₛ,ᵢ(x)
  absl::Span<const F> grand_product_evals;
  // Zₛ,ᵢ(ω * x)
  absl::Span<const F> grand_product_next_evals;
  const F& theta;
  const F& gamma;
};

}  // namespace tachyon::zk::shuffle

#endif  // TACHYON_ZK_SHUFFLE_VERIFIER_DATA_H_
