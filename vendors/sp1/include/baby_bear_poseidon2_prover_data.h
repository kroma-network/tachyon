#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_H_

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_constants.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

struct TachyonBabyBear;

class ProverData {
 public:
  ProverData() = default;
  ProverData(tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree,
             bool owned_tree)
      : tree_(tree), owned_tree_(owned_tree) {}
  ProverData(const ProverData& other) = delete;
  ProverData& operator=(const ProverData& other) = delete;
  ~ProverData();

  tachyon_baby_bear* commitment() { return commitment_; }
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree** tree_ptr() {
    return &tree_;
  }

  void write_commit(rust::Slice<TachyonBabyBear> values) const;
  std::unique_ptr<ProverData> clone() const;

 private:
  tachyon_baby_bear commitment_[TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK];
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree_ = nullptr;
  bool owned_tree_ = false;
};

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_PROVER_DATA_H_
