#include "vendors/sp1/include/baby_bear_poseidon2_two_adic_fri_pcs.h"

#include "vendors/sp1/src/baby_bear_poseidon2.rs.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

TwoAdicFriPcs::TwoAdicFriPcs(size_t log_blowup, size_t num_queries,
                             size_t proof_of_work_bits)
    : pcs_(tachyon_sp1_baby_bear_poseidon2_two_adic_fri_create(
          static_cast<uint32_t>(log_blowup), num_queries, proof_of_work_bits)) {
}

TwoAdicFriPcs::~TwoAdicFriPcs() {
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_destroy(pcs_);
}

void TwoAdicFriPcs::allocate_ldes(size_t size) const {
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_allocate_ldes(
      const_cast<tachyon_sp1_baby_bear_poseidon2_two_adic_fri*>(pcs_), size);
}

rust::Slice<TachyonBabyBear> TwoAdicFriPcs::coset_lde_batch(
    rust::Slice<TachyonBabyBear> values, size_t cols,
    const TachyonBabyBear& shift) const {
  size_t new_rows;
  tachyon_baby_bear* data =
      tachyon_sp1_baby_bear_poseidon2_two_adic_fri_coset_lde_batch(
          const_cast<tachyon_sp1_baby_bear_poseidon2_two_adic_fri*>(pcs_),
          reinterpret_cast<tachyon_baby_bear*>(values.data()),
          values.size() / cols, cols,
          reinterpret_cast<const tachyon_baby_bear&>(shift), &new_rows);
  return {reinterpret_cast<TachyonBabyBear*>(data), new_rows * cols};
}

std::unique_ptr<ProverData> TwoAdicFriPcs::commit() const {
  std::unique_ptr<ProverData> ret(new ProverData);
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_commit(
      const_cast<tachyon_sp1_baby_bear_poseidon2_two_adic_fri*>(pcs_),
      ret->commitment(), ret->tree_ptr());
  return ret;
}

std::unique_ptr<TwoAdicFriPcs> new_two_adic_fri_pcs(size_t log_blowup,
                                                    size_t num_queries,
                                                    size_t proof_of_work_bits) {
  return std::make_unique<TwoAdicFriPcs>(log_blowup, num_queries,
                                         proof_of_work_bits);
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
