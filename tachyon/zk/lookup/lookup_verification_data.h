#ifndef TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_DATA_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_DATA_H_

#include "tachyon/zk/plonk/vanishing/vanishing_verification_data.h"

namespace tachyon::zk {

template <typename F>
struct LookupVerificationData : public VanishingVerificationData<F> {
  F product_eval;
  F product_next_eval;
  F permuted_input_eval;
  F permuted_input_inv_eval;
  F permuted_table_eval;
  F theta;
  F beta;
  F gamma;
  F x;
  F l_first;
  F l_blind;
  F l_last;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_DATA_H_
