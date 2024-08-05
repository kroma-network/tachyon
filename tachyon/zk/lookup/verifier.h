#ifndef TACHYON_ZK_LOOKUP_VERIFIER_H_
#define TACHYON_ZK_LOOKUP_VERIFIER_H_

#include <type_traits>

#include "tachyon/zk/lookup/halo2/verifier.h"
#include "tachyon/zk/lookup/log_derivative_halo2/verifier.h"
#include "tachyon/zk/lookup/type.h"

namespace tachyon::zk::lookup {

template <Type kType, typename F, typename C>
using Verifier =
    std::conditional_t<kType == Type::kHalo2, halo2::Verifier<F, C>,
                       log_derivative_halo2::Verifier<F, C>>;

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_VERIFIER_H_
