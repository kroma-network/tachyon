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

void TwoAdicFriPcs::coset_lde_batch(
    rust::Slice<TachyonBabyBear> values, size_t cols,
    rust::Slice<TachyonBabyBear> extended_values,
    const TachyonBabyBear& shift) const {
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_coset_lde_batch(
      const_cast<tachyon_sp1_baby_bear_poseidon2_two_adic_fri*>(pcs_),
      reinterpret_cast<tachyon_baby_bear*>(values.data()), values.size() / cols,
      cols, reinterpret_cast<tachyon_baby_bear*>(extended_values.data()),
      reinterpret_cast<const tachyon_baby_bear&>(shift));
}

std::unique_ptr<ProverData> TwoAdicFriPcs::commit(
    const ProverDataVec& prover_data_vec) const {
  std::unique_ptr<ProverData> ret(new ProverData);
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_commit(
      const_cast<tachyon_sp1_baby_bear_poseidon2_two_adic_fri*>(pcs_),
      ret->commitment(), ret->tree_ptr(),
      const_cast<ProverDataVec&>(prover_data_vec).tree_vec());
  return ret;
}

std::unique_ptr<OpeningProof> TwoAdicFriPcs::do_open(
    const ProverDataVec& prover_data_vec, const OpeningPoints& opening_points,
    DuplexChallenger& challenger) const {
  std::unique_ptr<OpeningProof> ret(new OpeningProof);
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_open(
      pcs_, prover_data_vec.tree_vec(), opening_points.opening_points(),
      challenger.challenger(), ret->opened_values_ptr(), ret->proof_ptr());
  return ret;
}

bool TwoAdicFriPcs::do_verify(const CommitmentVec& commitment_vec,
                              const Domains& domains,
                              const OpeningPoints& opening_points,
                              const OpenedValues& opened_values,
                              const FriProof& proof,
                              DuplexChallenger& challenger) const {
  return tachyon_sp1_baby_bear_poseidon2_two_adic_fri_verify(
      pcs_, commitment_vec.commitment_vec(), domains.domains(),
      opening_points.opening_points(), opened_values.opened_values(),
      proof.proof(), challenger.challenger());
}

std::unique_ptr<TwoAdicFriPcs> new_two_adic_fri_pcs(size_t log_blowup,
                                                    size_t num_queries,
                                                    size_t proof_of_work_bits) {
  return std::make_unique<TwoAdicFriPcs>(log_blowup, num_queries,
                                         proof_of_work_bits);
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
