#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

class CommitResult;
class ProverData;
struct TachyonBabyBear;

class TwoAdicFriPcs {
 public:
  TwoAdicFriPcs(size_t log_blowup, size_t num_queries,
                size_t proof_of_work_bits);
  TwoAdicFriPcs(const TwoAdicFriPcs& other) = delete;
  TwoAdicFriPcs& operator=(const TwoAdicFriPcs& other) = delete;
  ~TwoAdicFriPcs();

  void allocate_ldes(size_t size) const;
  rust::Slice<TachyonBabyBear> coset_lde_batch(
      rust::Slice<TachyonBabyBear> values, size_t cols,
      const TachyonBabyBear& shift) const;
  std::unique_ptr<ProverData> commit() const;

 private:
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs_;
};

std::unique_ptr<TwoAdicFriPcs> new_two_adic_fri_pcs(size_t log_blowup,
                                                    size_t num_queries,
                                                    size_t proof_of_work_bits);

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_
