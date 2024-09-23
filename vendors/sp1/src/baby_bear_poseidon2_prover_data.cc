#include "vendors/sp1/include/baby_bear_poseidon2_prover_data.h"

#include <string.h>

#include "tachyon/base/logging.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_type_traits.h"
#include "vendors/sp1/src/baby_bear_poseidon2.rs.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

ProverData::~ProverData() {
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_destroy(tree_);
}

bool ProverData::eq(const ProverData& other) const {
  return c::base::native_cast(*tree_) == c::base::native_cast(*other.tree_);
}

void ProverData::write_commit(rust::Slice<TachyonBabyBear> values) const {
  CHECK_EQ(values.size(), size_t{TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK});
  for (size_t i = 0; i < values.size(); ++i) {
    memcpy(&values[i], &commitment_[i], sizeof(uint32_t));
  }
}

std::unique_ptr<ProverData> ProverData::clone() const {
  return std::make_unique<ProverData>(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_clone(tree_));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
