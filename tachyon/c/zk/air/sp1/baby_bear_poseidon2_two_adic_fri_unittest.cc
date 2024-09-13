#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/bits.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear4_type_traits.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/math/matrix/baby_bear_row_major_matrix_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_commitment_vec_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_domains_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opening_points_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri_type_traits.h"

namespace tachyon {

namespace {

using F = math::BabyBear;
using ExtF = math::BabyBear4;
using MMCS = c::zk::air::plonky3::baby_bear::MMCS;
using Coset = c::zk::air::plonky3::baby_bear::Coset;
using PCS = c::zk::air::plonky3::baby_bear::PCS;

constexpr uint32_t kLogBlowup = 1;
constexpr size_t kRounds = 1;

class TwoAdicFRITest : public testing::Test {
 public:
  void SetUp() override {
    pcs_ =
        tachyon_sp1_baby_bear_poseidon2_two_adic_fri_create(kLogBlowup, 10, 8);
    lde_vec_ = tachyon_sp1_baby_bear_poseidon2_lde_vec_create();
    prover_data_vec_ =
        tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_create(kRounds);
    opening_points_ =
        tachyon_sp1_baby_bear_poseidon2_opening_points_create(kRounds);
    challenger_ = tachyon_sp1_baby_bear_poseidon2_duplex_challenger_create();
    opened_values_ = tachyon_sp1_baby_bear_poseidon2_opened_values_create(0);
    proof_ = tachyon_sp1_baby_bear_poseidon2_fri_proof_create();
    commitment_vec_ =
        tachyon_sp1_baby_bear_poseidon2_commitment_vec_create(kRounds);
    domains_ = tachyon_sp1_baby_bear_poseidon2_domains_create(kRounds);
  }

  void TearDown() override {
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri_destroy(pcs_);
    tachyon_sp1_baby_bear_poseidon2_lde_vec_destroy(lde_vec_);
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_destroy(
        prover_data_vec_);
    tachyon_sp1_baby_bear_poseidon2_opening_points_destroy(opening_points_);
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger_destroy(challenger_);
    tachyon_sp1_baby_bear_poseidon2_opened_values_destroy(opened_values_);
    tachyon_sp1_baby_bear_poseidon2_fri_proof_destroy(proof_);
    tachyon_sp1_baby_bear_poseidon2_commitment_vec_destroy(commitment_vec_);
    tachyon_sp1_baby_bear_poseidon2_domains_destroy(domains_);
  }

 protected:
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs_ = nullptr;
  tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec_ = nullptr;
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* prover_data_vec_ =
      nullptr;
  tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points_ = nullptr;
  tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger_ = nullptr;
  tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values_ = nullptr;
  tachyon_sp1_baby_bear_poseidon2_fri_proof* proof_ = nullptr;
  tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitment_vec_ = nullptr;
  tachyon_sp1_baby_bear_poseidon2_domains* domains_ = nullptr;
};

}  // namespace

TEST_F(TwoAdicFRITest, APIs) {
  constexpr size_t kRowsForInput = 32;
  constexpr size_t kColsForInput = 5;
  constexpr size_t kExtendedRowsForInput = kRowsForInput << kLogBlowup;
  constexpr size_t kRowsForOpening = 2;
  constexpr size_t kColsForOpening = 2;

  using Commitment = typename MMCS::Commitment;
  using ProverData = typename MMCS::ProverData;

  tachyon_sp1_baby_bear_poseidon2_two_adic_fri* another_pcs =
      tachyon_sp1_baby_bear_poseidon2_two_adic_fri_create(1, 10, 8);
  PCS* native_pcs = c::base::native_cast(another_pcs);

  Coset coset = native_pcs->GetNaturalDomainForDegree(kRowsForInput);

  std::vector<F> matrix_data = base::CreateVector(kRowsForInput * kColsForInput,
                                                  []() { return F::Random(); });
  std::vector<F> matrix_data_clone = matrix_data;

  std::vector<F> extended_matrix_data(kExtendedRowsForInput * kColsForInput);
  F shift = F::FromMontgomery(F::Config::kSubgroupGenerator) *
            coset.domain()->offset_inv();
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_coset_lde_batch(
      pcs_, c::base::c_cast(const_cast<F*>(matrix_data.data())), kRowsForInput,
      kColsForInput,
      c::base::c_cast(const_cast<F*>(extended_matrix_data.data())),
      c::base::c_cast(shift));
  tachyon_sp1_baby_bear_poseidon2_lde_vec_add(
      lde_vec_, c::base::c_cast(const_cast<F*>(extended_matrix_data.data())),
      kExtendedRowsForInput, kColsForInput);
  tachyon_baby_bear commitment[TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK];
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* prover_data = nullptr;
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_commit(pcs_, lde_vec_,
                                                      commitment, &prover_data);
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_set(prover_data_vec_, 0,
                                                            prover_data);

  Commitment native_commitment;
  ProverData native_prover_data;
  std::vector<Coset> cosets;
  cosets.push_back(std::move(coset));
  std::vector<math::RowMajorMatrix<F>> matrices;
  matrices.push_back(Eigen::Map<const math::RowMajorMatrix<F>>(
      matrix_data_clone.data(), kRowsForInput, kColsForInput));
  ASSERT_TRUE(native_pcs->Commit(cosets, matrices, &native_commitment,
                                 &native_prover_data));

  for (size_t i = 0; i < TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK; ++i) {
    EXPECT_EQ(c::base::native_cast(commitment[i]), native_commitment[i]);
  }

  math::RowMajorMatrix<F> leaf = Eigen::Map<math::RowMajorMatrix<F>>(
      extended_matrix_data.data(), kExtendedRowsForInput, kColsForInput);
  EXPECT_EQ(leaf, native_prover_data.leaves()[0]);

  ExtF point = ExtF::Random();

  tachyon_sp1_baby_bear_poseidon2_opening_points_allocate(
      opening_points_, 0, kRowsForOpening, kColsForOpening);
  for (size_t r = 0; r < kRowsForOpening; ++r) {
    for (size_t c = 0; c < kColsForOpening; ++c) {
      tachyon_sp1_baby_bear_poseidon2_opening_points_set(
          opening_points_, 0, r, c, c::base::c_cast(&point));
    }
  }

  tachyon_sp1_baby_bear_poseidon2_duplex_challenger* another_challenger =
      tachyon_sp1_baby_bear_poseidon2_duplex_challenger_clone(challenger_);
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_open(
      pcs_, prover_data_vec_, opening_points_, challenger_, &opened_values_,
      &proof_);

  tachyon_sp1_baby_bear_poseidon2_commitment_vec_set(commitment_vec_, 0,
                                                     commitment);
  tachyon_sp1_baby_bear_poseidon2_domains_allocate(domains_, 0, 1);
  tachyon_sp1_baby_bear_poseidon2_domains_set(
      domains_, 0, 0, base::bits::CheckedLog2(kRowsForInput),
      c::base::c_cast(&shift));

  ASSERT_TRUE(tachyon_sp1_baby_bear_poseidon2_two_adic_fri_verify(
      pcs_, commitment_vec_, domains_, opening_points_, opened_values_, proof_,
      another_challenger));

  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_destroy(another_pcs);
  tachyon_sp1_baby_bear_poseidon2_duplex_challenger_destroy(another_challenger);
}

}  // namespace tachyon
