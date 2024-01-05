#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_DATA_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_DATA_H_

#include <optional>

#include "tachyon/zk/plonk/vanishing/vanishing_verification_data.h"

namespace tachyon::zk {

template <typename F>
struct PermutationVerificationData : public VanishingVerificationData<F> {
  absl::Span<const F> common_evals;
  absl::Span<const F> product_evals;
  absl::Span<const F> product_next_evals;
  absl::Span<const std::optional<F>> product_last_evals;
  F beta;
  F gamma;
  F x;
  F l_first;
  F l_blind;
  F l_last;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_DATA_H_
