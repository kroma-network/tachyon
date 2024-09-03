#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_VEC_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_VEC_H_

#include <memory>

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

struct TachyonBabyBear;

class ProverDataVec {
 public:
  explicit ProverDataVec(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec)
      : tree_vec_(tree_vec) {}
  ProverDataVec(const ProverDataVec& other) = delete;
  ProverDataVec& operator=(const ProverDataVec& other) = delete;
  ~ProverDataVec();

  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec() {
    return tree_vec_;
  }
  const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec()
      const {
    return tree_vec_;
  }

  std::unique_ptr<ProverDataVec> clone() const;

 private:
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec_ = nullptr;
};

std::unique_ptr<ProverDataVec> new_prover_data_vec();

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_VEC_H_
