#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_FRI_PROOF_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_FRI_PROOF_H_

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

class FriProof {
 public:
  explicit FriProof(tachyon_sp1_baby_bear_poseidon2_fri_proof* proof)
      : proof_(proof) {}
  FriProof(const FriProof& other) = delete;
  FriProof& operator=(const FriProof& other) = delete;
  ~FriProof();

  std::unique_ptr<FriProof> clone() const;

 private:
  tachyon_sp1_baby_bear_poseidon2_fri_proof* proof_;
};

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_FRI_PROOF_H_
