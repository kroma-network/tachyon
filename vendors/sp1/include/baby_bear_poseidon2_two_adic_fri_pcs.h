#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

class CommitResult;
class DuplexChallenger;
class OpeningPoints;
class OpeningProof;
class ProverData;
class ProverDataVec;
struct TachyonBabyBear;
struct TachyonBabyBear4;

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
  std::unique_ptr<ProverData> commit(
      const ProverDataVec& prover_data_vec) const;
  std::unique_ptr<OpeningProof> do_open(const ProverDataVec& prover_data_vec,
                                        const OpeningPoints& opening_points,
                                        DuplexChallenger& challenger) const;

 private:
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs_;
};

std::unique_ptr<TwoAdicFriPcs> new_two_adic_fri_pcs(size_t log_blowup,
                                                    size_t num_queries,
                                                    size_t proof_of_work_bits);

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_PCS_H_
