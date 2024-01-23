#ifndef TACHYON_ZK_PLONK_HALO2_STEP_RETURNS_H_
#define TACHYON_ZK_PLONK_HALO2_STEP_RETURNS_H_

#include <utility>
#include <vector>

namespace tachyon::zk::halo2 {

template <typename P, typename L, typename V>
class StepReturns {
 public:
  StepReturns(std::vector<P>&& permutations,
              std::vector<std::vector<L>>&& lookups_vec, V&& vanishing)
      : permutations_(std::move(permutations)),
        lookups_vec_(std::move(lookups_vec)),
        vanishing_(std::move(vanishing)) {}

  const std::vector<P>& permutations() const { return permutations_; }
  const std::vector<std::vector<L>>& lookups_vec() const {
    return lookups_vec_;
  }
  const V& vanishing() const { return vanishing_; }

  std::vector<P>&& TakePermutations() && { return std::move(permutations_); }
  std::vector<std::vector<L>>&& TakeLookupsVec() && {
    return std::move(lookups_vec_);
  }
  V&& TakeVanishing() && { return std::move(vanishing_); }

 private:
  std::vector<P> permutations_;
  std::vector<std::vector<L>> lookups_vec_;
  V vanishing_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_STEP_RETURNS_H_
