#ifndef TACHYON_ZK_LOOKUP_PROVER_H_
#define TACHYON_ZK_LOOKUP_PROVER_H_

#include <type_traits>

#include "tachyon/zk/lookup/halo2/prover.h"
#include "tachyon/zk/lookup/log_derivative_halo2/prover.h"
#include "tachyon/zk/lookup/type.h"

namespace tachyon::zk::lookup {

template <Type kType, typename Poly, typename Evals>
using Prover =
    std::conditional_t<kType == Type::kHalo2, halo2::Prover<Poly, Evals>,
                       log_derivative_halo2::Prover<Poly, Evals>>;

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_PROVER_H_
