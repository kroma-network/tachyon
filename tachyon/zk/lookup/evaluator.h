#ifndef TACHYON_ZK_LOOKUP_EVALUATOR_H_
#define TACHYON_ZK_LOOKUP_EVALUATOR_H_

#include <type_traits>

#include "tachyon/zk/lookup/halo2/evaluator.h"
#include "tachyon/zk/lookup/log_derivative_halo2/evaluator.h"
#include "tachyon/zk/lookup/type.h"

namespace tachyon::zk::lookup {

template <Type kType, typename EvalsOrExtendedEvals>
using Evaluator = std::conditional_t<
    kType == lookup::Type::kHalo2,
    lookup::halo2::Evaluator<EvalsOrExtendedEvals>,
    lookup::log_derivative_halo2::Evaluator<EvalsOrExtendedEvals>>;

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_EVALUATOR_H_
