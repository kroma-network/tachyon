#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENING_PROOF_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENING_PROOF_H_

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

class FriProof;

class OpeningProof {
 public:
  OpeningProof();
  OpeningProof(const OpeningProof& other) = delete;
  OpeningProof& operator=(const OpeningProof& other) = delete;
  ~OpeningProof();

  tachyon_sp1_baby_bear_poseidon2_opened_values** opened_values_ptr() {
    return &opened_values_;
  }
  tachyon_sp1_baby_bear_poseidon2_fri_proof** proof_ptr() { return &proof_; }

  rust::Vec<uint8_t> serialize_to_opened_values() const;
  std::unique_ptr<FriProof> take_fri_proof();

 private:
  tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values_;
  tachyon_sp1_baby_bear_poseidon2_fri_proof* proof_;
};

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENING_PROOF_H_
