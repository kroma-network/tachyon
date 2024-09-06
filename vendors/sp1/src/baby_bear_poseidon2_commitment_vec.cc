#include "vendors/sp1/include/baby_bear_poseidon2_commitment_vec.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

CommitmentVec::~CommitmentVec() {
  tachyon_sp1_baby_bear_poseidon2_commitment_vec_destroy(commitment_vec_);
}

void CommitmentVec::set(size_t round,
                        rust::Slice<const TachyonBabyBear> commitment) {
  tachyon_sp1_baby_bear_poseidon2_commitment_vec_set(
      commitment_vec_, round,
      reinterpret_cast<const tachyon_baby_bear*>(commitment.data()));
}

std::unique_ptr<CommitmentVec> new_commitment_vec(size_t rounds) {
  return std::make_unique<CommitmentVec>(
      tachyon_sp1_baby_bear_poseidon2_commitment_vec_create(rounds));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
