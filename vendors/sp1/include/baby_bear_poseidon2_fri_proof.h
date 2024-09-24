#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_FRI_PROOF_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_FRI_PROOF_H_

#include <stdint.h>

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

  const tachyon_sp1_baby_bear_poseidon2_fri_proof* proof() const {
    return proof_;
  }

  bool eq(const FriProof& other) const;
  rust::Vec<uint8_t> write_hint() const;
  rust::Vec<uint8_t> serialize() const;
  std::unique_ptr<FriProof> clone() const;

 private:
  tachyon_sp1_baby_bear_poseidon2_fri_proof* proof_;
};

std::unique_ptr<FriProof> deserialize_fri_proof(
    rust::Slice<const uint8_t> data);

std::unique_ptr<FriProof> deserialize_json_fri_proof(
    rust::Slice<const uint8_t> data);

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_FRI_PROOF_H_
