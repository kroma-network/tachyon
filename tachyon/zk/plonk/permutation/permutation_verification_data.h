#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_DATA_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_DATA_H_

#include <optional>

#include "tachyon/zk/plonk/vanishing/vanishing_verification_data.h"

namespace tachyon::zk::plonk {

template <typename F, typename C>
struct PermutationVerificationData : public VanishingVerificationData<F> {
  absl::Span<const C> product_commitments;
  absl::Span<const F> common_evals;
  absl::Span<const F> product_evals;
  absl::Span<const F> product_next_evals;
  absl::Span<const std::optional<F>> product_last_evals;
  const F* beta = nullptr;
  const F* gamma = nullptr;
  const F* x = nullptr;
  const F* x_next = nullptr;
  const F* x_last = nullptr;
  const F* l_first = nullptr;
  const F* l_blind = nullptr;
  const F* l_last = nullptr;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFICATION_DATA_H_
