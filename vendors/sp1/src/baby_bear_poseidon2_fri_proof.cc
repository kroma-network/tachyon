#include "vendors/sp1/include/baby_bear_poseidon2_fri_proof.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

FriProof::~FriProof() {
  tachyon_sp1_baby_bear_poseidon2_fri_proof_destroy(proof_);
}

rust::Vec<uint8_t> FriProof::write_hint() const {
  rust::Vec<uint8_t> ret;
  size_t size;
  tachyon_sp1_baby_bear_poseidon2_fri_proof_write_hint(proof_, nullptr, &size);
  // NOTE(chokobole): |rust::Vec<uint8_t>| doesn't have |resize()|.
  ret.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    ret.push_back(0);
  }
  tachyon_sp1_baby_bear_poseidon2_fri_proof_write_hint(proof_, ret.data(),
                                                       &size);
  return ret;
}

std::unique_ptr<FriProof> FriProof::clone() const {
  return std::make_unique<FriProof>(
      tachyon_sp1_baby_bear_poseidon2_fri_proof_clone(proof_));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
