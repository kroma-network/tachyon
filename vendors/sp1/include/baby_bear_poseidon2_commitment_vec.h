#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_H_

#include <stddef.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_commitment_vec.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

struct TachyonBabyBear;

class CommitmentVec {
 public:
  explicit CommitmentVec(
      tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitment_vec)
      : commitment_vec_(commitment_vec) {}
  CommitmentVec(const CommitmentVec& other) = delete;
  CommitmentVec& operator=(const CommitmentVec& other) = delete;
  ~CommitmentVec();

  const tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitment_vec() const {
    return commitment_vec_;
  }

  void set(size_t round, rust::Slice<const TachyonBabyBear> commitment);

 private:
  tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitment_vec_;
};

std::unique_ptr<CommitmentVec> new_commitment_vec(size_t rounds);

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_COMMITMENT_VEC_H_
