#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFICATION_DATA_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFICATION_DATA_H_

#include "absl/types/span.h"

namespace tachyon::zk {

template <typename F>
struct VanishingVerificationData {
  absl::Span<const F> fixed_evals;
  absl::Span<const F> advice_evals;
  absl::Span<const F> instance_evals;
  absl::Span<const F> challenges;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_VERIFICATION_DATA_H_
