#include "vendors/sp1/include/baby_bear_poseidon2_prover_data_vec.h"

#include "vendors/sp1/src/baby_bear_poseidon2.rs.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

ProverDataVec::~ProverDataVec() {
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_destroy(tree_vec_);
}

std::unique_ptr<ProverDataVec> ProverDataVec::clone() const {
  return std::make_unique<ProverDataVec>(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_clone(tree_vec_));
}

void ProverDataVec::set(size_t round, const ProverData& prover_data) {
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_set(tree_vec_, round,
                                                            prover_data.tree());
}

std::unique_ptr<ProverDataVec> new_prover_data_vec(size_t rounds) {
  return std::make_unique<ProverDataVec>(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_create(rounds));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
