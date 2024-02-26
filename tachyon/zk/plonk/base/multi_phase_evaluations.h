#ifndef TACHYON_ZK_PLONK_BASE_MULTI_PHASE_EVALUATIONS_H_
#define TACHYON_ZK_PLONK_BASE_MULTI_PHASE_EVALUATIONS_H_

#include "absl/types/span.h"

namespace tachyon::zk::plonk {

template <typename F>
struct MultiPhaseEvaluations {
  MultiPhaseEvaluations() = default;
  MultiPhaseEvaluations(absl::Span<const F> fixed_evals,
                        absl::Span<const F> advice_evals,
                        absl::Span<const F> instance_evals,
                        absl::Span<const F> challenges)
      : fixed_evals(fixed_evals),
        advice_evals(advice_evals),
        instance_evals(instance_evals),
        challenges(challenges) {}

  absl::Span<const F> fixed_evals;
  absl::Span<const F> advice_evals;
  absl::Span<const F> instance_evals;
  absl::Span<const F> challenges;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_BASE_MULTI_PHASE_EVALUATIONS_H_
