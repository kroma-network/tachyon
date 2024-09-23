#include "vendors/sp1/include/baby_bear_poseidon2_fri_proof.h"

#include "tachyon/rs/base/container_util.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

FriProof::~FriProof() {
  tachyon_sp1_baby_bear_poseidon2_fri_proof_destroy(proof_);
}

rust::Vec<uint8_t> FriProof::write_hint() const {
  size_t size;
  tachyon_sp1_baby_bear_poseidon2_fri_proof_write_hint(proof_, nullptr, &size);
  rust::Vec<uint8_t> ret = rs::CreateEmptyVector<uint8_t>(size);
  tachyon_sp1_baby_bear_poseidon2_fri_proof_write_hint(proof_, ret.data(),
                                                       &size);
  return ret;
}

std::unique_ptr<FriProof> FriProof::clone() const {
  return std::make_unique<FriProof>(
      tachyon_sp1_baby_bear_poseidon2_fri_proof_clone(proof_));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
