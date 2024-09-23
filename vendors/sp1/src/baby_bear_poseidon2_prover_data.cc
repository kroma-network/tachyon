#include "vendors/sp1/include/baby_bear_poseidon2_prover_data.h"

#include <string.h>

#include "tachyon/base/logging.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_type_traits.h"
#include "tachyon/rs/base/container_util.h"
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

rust::Vec<uint8_t> ProverData::serialize() const {
  size_t size;
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_serialize(tree_, nullptr,
                                                              &size);
  rust::Vec<uint8_t> ret = rs::CreateEmptyVector<uint8_t>(size);
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_serialize(tree_, ret.data(),
                                                              &size);
  return ret;
}

std::unique_ptr<ProverData> ProverData::clone() const {
  return std::make_unique<ProverData>(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_clone(tree_));
}

std::unique_ptr<ProverData> deserialize_prover_data(
    rust::Slice<const uint8_t> data) {
  return std::make_unique<ProverData>(
      tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_deserialize(
          data.data(), data.size()));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
