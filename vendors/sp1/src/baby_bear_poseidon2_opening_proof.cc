#include "vendors/sp1/include/baby_bear_poseidon2_opening_proof.h"

#include <utility>

#include "vendors/sp1/src/baby_bear_poseidon2.rs.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

OpeningProof::OpeningProof()
    : opened_values_(tachyon_sp1_baby_bear_poseidon2_opened_values_create()),
      proof_(tachyon_sp1_baby_bear_poseidon2_fri_proof_create()) {}

OpeningProof::~OpeningProof() {
  tachyon_sp1_baby_bear_poseidon2_opened_values_destroy(opened_values_);
  if (proof_) {
    tachyon_sp1_baby_bear_poseidon2_fri_proof_destroy(proof_);
  }
}

rust::Vec<uint8_t> OpeningProof::serialize_to_opened_values() const {
  size_t size;
  tachyon_sp1_baby_bear_poseidon2_opened_values_serialize(opened_values_,
                                                          nullptr, &size);
  rust::Vec<uint8_t> ret;
  ret.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    ret.push_back(0);
  }
  tachyon_sp1_baby_bear_poseidon2_opened_values_serialize(opened_values_,
                                                          ret.data(), &size);
  return ret;
}

std::unique_ptr<FriProof> OpeningProof::take_fri_proof() {
  return std::make_unique<FriProof>(std::exchange(proof_, nullptr));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
