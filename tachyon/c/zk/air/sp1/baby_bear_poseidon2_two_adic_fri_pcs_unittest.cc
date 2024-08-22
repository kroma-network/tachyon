#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri_pcs.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/math/matrix/baby_bear_row_major_matrix_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri_pcs_type_traits.h"

namespace tachyon {

namespace {

using F = math::BabyBear;
using MMCS = c::zk::air::plonky3::baby_bear::MMCS;
using Coset = c::zk::air::plonky3::baby_bear::Coset;
using PCS = c::zk::air::plonky3::baby_bear::PCS;

class TwoAdicFriPCSTest : public testing::Test {
 public:
  void SetUp() override {
    pcs_ = tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_create(1, 10, 8);
  }

  void TearDown() override {
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_destroy(pcs_);
  }

 protected:
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs* pcs_ = nullptr;
};

}  // namespace

TEST_F(TwoAdicFriPCSTest, APIs) {
  constexpr size_t kRows = 32;
  constexpr size_t kCols = 5;

  using Commitment = typename MMCS::Commitment;
  using ProverData = typename MMCS::ProverData;

  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs* another_pcs =
      tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_create(1, 10, 8);
  PCS* native_pcs = c::base::native_cast(another_pcs);

  Coset coset = native_pcs->GetNaturalDomainForDegree(kRows);

  std::vector<F> matrix_data =
      base::CreateVector(kRows * kCols, []() { return F::Random(); });
  std::vector<F> matrix_data_clone = matrix_data;

  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_allocate_ldes(pcs_, 1);
  F shift = F::FromMontgomery(F::Config::kSubgroupGenerator) *
            coset.domain()->offset_inv();
  size_t new_rows;
  tachyon_baby_bear* new_matrix_data =
      tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_coset_lde_batch(
          pcs_, c::base::c_cast(const_cast<F*>(matrix_data.data())), kRows,
          kCols, c::base::c_cast(shift), &new_rows);
  tachyon_baby_bear commitment[TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK];
  tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* prover_data = nullptr;
  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_commit(pcs_, commitment,
                                                          &prover_data);

  Commitment native_commitment;
  ProverData native_prover_data;
  std::vector<Coset> cosets;
  cosets.push_back(std::move(coset));
  std::vector<math::RowMajorMatrix<F>> matrices;
  matrices.push_back(Eigen::Map<const math::RowMajorMatrix<F>>(
      matrix_data_clone.data(), kRows, kCols));
  ASSERT_TRUE(native_pcs->Commit(cosets, matrices, &native_commitment,
                                 &native_prover_data));

  for (size_t i = 0; i < TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK; ++i) {
    EXPECT_EQ(c::base::native_cast(commitment[i]), native_commitment[i]);
  }

  math::RowMajorMatrix<F> leaf = Eigen::Map<math::RowMajorMatrix<F>>(
      c::base::native_cast(new_matrix_data), new_rows, kCols);
  EXPECT_EQ(leaf, native_prover_data.leaves()[0]);

  tachyon_sp1_baby_bear_poseidon2_two_adic_fri_pcs_destroy(another_pcs);
}

}  // namespace tachyon
