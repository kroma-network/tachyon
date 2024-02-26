#ifndef TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_DATA_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_DATA_H_

#include "tachyon/zk/lookup/lookup_pair.h"
#include "tachyon/zk/plonk/vanishing/vanishing_verification_data.h"

namespace tachyon::zk::lookup {

template <typename F, typename C>
struct VerificationData : public plonk::VanishingVerificationData<F> {
  const Pair<C>* permuted_commitment = nullptr;
  const C* product_commitment = nullptr;
  const F* product_eval = nullptr;
  const F* product_next_eval = nullptr;
  const F* permuted_input_eval = nullptr;
  const F* permuted_input_prev_eval = nullptr;
  const F* permuted_table_eval = nullptr;
  const F* theta = nullptr;
  const F* beta = nullptr;
  const F* gamma = nullptr;
  const F* x = nullptr;
  const F* x_next = nullptr;
  const F* x_prev = nullptr;
  const F* l_first = nullptr;
  const F* l_blind = nullptr;
  const F* l_last = nullptr;
};

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_DATA_H_
