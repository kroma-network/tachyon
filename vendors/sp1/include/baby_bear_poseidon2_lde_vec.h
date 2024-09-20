#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_LDE_VEC_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_LDE_VEC_H_

#include <stddef.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_lde_vec.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

struct TachyonBabyBear;

class LDEVec {
 public:
  explicit LDEVec(tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec)
      : lde_vec_(lde_vec) {}
  LDEVec(const LDEVec& other) = delete;
  LDEVec& operator=(const LDEVec& other) = delete;
  ~LDEVec();

  tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec() { return lde_vec_; }

  void add(rust::Slice<const TachyonBabyBear> lde, size_t cols);

 private:
  tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec_;
};

std::unique_ptr<LDEVec> new_lde_vec();

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_LDE_VEC_H_
