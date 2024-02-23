#ifndef TACHYON_ZK_LOOKUP_HALO2_VERIFIER_DATA_H_
#define TACHYON_ZK_LOOKUP_HALO2_VERIFIER_DATA_H_

#include "tachyon/zk/lookup/lookup_pair.h"
#include "tachyon/zk/plonk/base/multi_phase_evaluations.h"

namespace tachyon::zk::lookup::halo2 {

template <typename F, typename C>
struct VerifierData : public plonk::MultiPhaseEvaluations<F> {
  VerifierData(absl::Span<const F> fixed_evals,
               absl::Span<const F> advice_evals,
               absl::Span<const F> instance_evals,
               absl::Span<const F> challenges,
               absl::Span<const lookup::Pair<C>> permuted_commitments,
               absl::Span<const C> grand_product_commitments,
               absl::Span<const F> grand_product_evals,
               absl::Span<const F> grand_product_next_evals,
               absl::Span<const F> permuted_input_evals,
               absl::Span<const F> permuted_input_prev_evals,
               absl::Span<const F> permuted_table_evals, const F& theta,
               const F& beta, const F& gamma)
      : plonk::MultiPhaseEvaluations<F>(fixed_evals, advice_evals,
                                        instance_evals, challenges),
        permuted_commitments(permuted_commitments),
        grand_product_commitments(grand_product_commitments),
        grand_product_evals(grand_product_evals),
        grand_product_next_evals(grand_product_next_evals),
        permuted_input_evals(permuted_input_evals),
        permuted_input_prev_evals(permuted_input_prev_evals),
        permuted_table_evals(permuted_table_evals),
        theta(theta),
        beta(beta),
        gamma(gamma) {}

  // [{A'ᵢ(τ), S'ᵢ(τ)}]₁
  absl::Span<const lookup::Pair<C>> permuted_commitments;
  // [Zₗ,ᵢ(τ)]₁
  absl::Span<const C> grand_product_commitments;
  // Zₗ,ᵢ(x)
  absl::Span<const F> grand_product_evals;
  // Zₗ,ᵢ(ω * x)
  absl::Span<const F> grand_product_next_evals;
  // A'ᵢ(x)
  absl::Span<const F> permuted_input_evals;
  // A'ᵢ(ω⁻¹ * x)
  absl::Span<const F> permuted_input_prev_evals;
  // S'ᵢ(x)
  absl::Span<const F> permuted_table_evals;
  const F& theta;
  const F& beta;
  const F& gamma;
};

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_VERIFIER_DATA_H_
