#include "vendors/sp1/include/baby_bear_poseidon2_lde_vec.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

LDEVec::~LDEVec() { tachyon_sp1_baby_bear_poseidon2_lde_vec_destroy(lde_vec_); }

void LDEVec::add(rust::Slice<const TachyonBabyBear> lde, size_t cols) {
  tachyon_sp1_baby_bear_poseidon2_lde_vec_add(
      lde_vec_, reinterpret_cast<const tachyon_baby_bear*>(lde.data()),
      lde.size() / cols, cols);
}

std::unique_ptr<LDEVec> new_lde_vec() {
  return std::make_unique<LDEVec>(
      tachyon_sp1_baby_bear_poseidon2_lde_vec_create());
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
