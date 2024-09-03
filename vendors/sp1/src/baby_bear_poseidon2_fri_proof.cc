#include "vendors/sp1/include/baby_bear_poseidon2_fri_proof.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

FriProof::~FriProof() {
  tachyon_sp1_baby_bear_poseidon2_fri_proof_destroy(proof_);
}

std::unique_ptr<FriProof> FriProof::clone() const {
  return std::make_unique<FriProof>(
      tachyon_sp1_baby_bear_poseidon2_fri_proof_clone(proof_));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
