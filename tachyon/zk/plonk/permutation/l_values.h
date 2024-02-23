#ifndef TACHYON_ZK_PLONK_PERMUTATION_L_VALUES_H_
#define TACHYON_ZK_PLONK_PERMUTATION_L_VALUES_H_

namespace tachyon::zk::plonk {

template <typename F>
struct LValues {
  LValues(const F& first, const F& blind, const F& last)
      : first(first), blind(blind), last(last) {}

  const F& first;
  const F& blind;
  const F& last;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_L_VALUES_H_
