#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFIER_DATA_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFIER_DATA_H_

#include <optional>

#include "tachyon/zk/plonk/base/multi_phase_evaluations.h"

namespace tachyon::zk::plonk {

template <typename F, typename C>
struct PermutationVerifierData : public MultiPhaseEvaluations<F> {
  PermutationVerifierData(
      absl::Span<const F> fixed_evals, absl::Span<const F> advice_evals,
      absl::Span<const F> instance_evals, absl::Span<const F> challenges,
      absl::Span<const C> grand_product_commitments,
      absl::Span<const F> grand_product_evals,
      absl::Span<const F> grand_product_next_evals,
      absl::Span<const std::optional<F>> grand_product_last_evals,
      absl::Span<const C> substitution_commitments,
      absl::Span<const F> substitution_evals, const F& beta, const F& gamma)
      : MultiPhaseEvaluations<F>(fixed_evals, advice_evals, instance_evals,
                                 challenges),
        grand_product_commitments(grand_product_commitments),
        grand_product_evals(grand_product_evals),
        grand_product_next_evals(grand_product_next_evals),
        grand_product_last_evals(grand_product_last_evals),
        substitution_commitments(substitution_commitments),
        substitution_evals(substitution_evals),
        beta(beta),
        gamma(gamma) {}

  // [Zₚ,ᵢ(τ)]₁
  absl::Span<const C> grand_product_commitments;
  // Zₚ,ᵢ(x)
  absl::Span<const F> grand_product_evals;
  // Zₚ,ᵢ(ω * x)
  absl::Span<const F> grand_product_next_evals;
  // Zₚ,ᵢ(ω^(last) * x)
  absl::Span<const std::optional<F>> grand_product_last_evals;
  // [sᵢ(τ)]₁
  absl::Span<const C> substitution_commitments;
  // sᵢ(x)
  absl::Span<const F> substitution_evals;
  const F& beta;
  const F& gamma;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFIER_DATA_H_
