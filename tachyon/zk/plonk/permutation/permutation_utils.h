#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_UTILS_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_UTILS_H_

#include <stddef.h>

#include "tachyon/base/numerics/checked_math.h"

namespace tachyon::zk {

constexpr size_t ComputePermutationChunkLength(size_t cs_degree) {
  base::CheckedNumeric<size_t> checked_cs_degree(cs_degree);
  return (checked_cs_degree - 2).ValueOrDie();
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_UTILS_H_
