// Copyright (c) 2022-2024 Scroll
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.scroll and the LICENCE-APACHE.scroll
// file.

#ifndef TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_VERIFIER_DATA_H_
#define TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_VERIFIER_DATA_H_

#include "tachyon/zk/plonk/base/multi_phase_evaluations.h"

namespace tachyon::zk::lookup::log_derivative_halo2 {

template <typename F, typename C>
struct VerifierData : public plonk::MultiPhaseEvaluations<F> {
  VerifierData(absl::Span<const F> fixed_evals,
               absl::Span<const F> advice_evals,
               absl::Span<const F> instance_evals,
               absl::Span<const F> challenges,
               absl::Span<const C> m_poly_commitments,
               absl::Span<const C> grand_sum_commitments,
               absl::Span<const F> grand_sum_evals,
               absl::Span<const F> grand_sum_next_evals,
               absl::Span<const F> m_evals, const F& theta, const F& beta)
      : plonk::MultiPhaseEvaluations<F>(fixed_evals, advice_evals,
                                        instance_evals, challenges),
        m_poly_commitments(m_poly_commitments),
        grand_sum_commitments(grand_sum_commitments),
        grand_sum_evals(grand_sum_evals),
        grand_sum_next_evals(grand_sum_next_evals),
        m_evals(m_evals),
        theta(theta),
        beta(beta) {}

  // [m(τ)]₁
  absl::Span<const C> m_poly_commitments;
  // [ϕ(τ)]₁
  absl::Span<const C> grand_sum_commitments;
  // ϕ(X)
  absl::Span<const F> grand_sum_evals;
  // ϕ(ω * X)
  absl::Span<const F> grand_sum_next_evals;
  // m(X)
  absl::Span<const F> m_evals;
  const F& theta;
  const F& beta;
};

}  // namespace tachyon::zk::lookup::log_derivative_halo2

#endif  // TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_VERIFIER_DATA_H_
