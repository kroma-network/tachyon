#include "vendors/sp1/include/baby_bear_poseidon2_prover_data_vec.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

ProverDataVec::~ProverDataVec() {
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_destroy(tree_vec_);
}

std::unique_ptr<ProverDataVec> ProverDataVec::clone() const {
  return std::make_unique<ProverDataVec>(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_clone(tree_vec_));
}

std::unique_ptr<ProverDataVec> new_prover_data_vec() {
  return std::make_unique<ProverDataVec>(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_create());
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
