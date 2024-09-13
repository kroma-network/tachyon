#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_VEC_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_VEC_H_

#include <stddef.h>

#include <memory>

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

class ProverData;
struct TachyonBabyBear;

class ProverDataVec {
 public:
  explicit ProverDataVec(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec)
      : tree_vec_(tree_vec) {}
  ProverDataVec(const ProverDataVec& other) = delete;
  ProverDataVec& operator=(const ProverDataVec& other) = delete;
  ~ProverDataVec();

  const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec()
      const {
    return tree_vec_;
  }

  std::unique_ptr<ProverDataVec> clone() const;
  void set(size_t round, const ProverData& prover_data);

 private:
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec_ = nullptr;
};

std::unique_ptr<ProverDataVec> new_prover_data_vec(size_t rounds);

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_VEC_H_
