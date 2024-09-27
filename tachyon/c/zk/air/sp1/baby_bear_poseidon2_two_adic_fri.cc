#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri.h"

#include <utility>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear4_type_traits.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/math/matrix/baby_bear_row_major_matrix_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_commitment_vec_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_domains_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_lde_vec_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opening_points_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri_type_traits.h"

using namespace tachyon;

using F = math::BabyBear;
using PackedF = math::PackedBabyBear;
using ExtF = math::BabyBear4;
using ExtPackedF = math::PackedBabyBear4;
using Poseidon2 = c::zk::air::sp1::baby_bear::Poseidon2;
using PackedPoseidon2 = c::zk::air::sp1::baby_bear::PackedPoseidon2;
using Hasher = c::zk::air::sp1::baby_bear::Hasher;
using PackedHasher = c::zk::air::sp1::baby_bear::PackedHasher;
using Compressor = c::zk::air::sp1::baby_bear::Compressor;
using PackedCompressor = c::zk::air::sp1::baby_bear::PackedCompressor;
using Tree = c::zk::air::sp1::baby_bear::Tree;
using MMCS = c::zk::air::sp1::baby_bear::MMCS;
using ExtMMCS = c::zk::air::sp1::baby_bear::ExtMMCS;
using ChallengeMMCS = c::zk::air::sp1::baby_bear::ChallengeMMCS;
using PCS = c::zk::air::sp1::baby_bear::PCS;
using Params =
    crypto::Poseidon2Params<crypto::Poseidon2Vendor::kPlonky3,
                            crypto::Poseidon2Vendor::kPlonky3, F,
                            TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1,
                            TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_ALPHA>;
using PackedParams =
    crypto::Poseidon2Params<crypto::Poseidon2Vendor::kPlonky3,
                            crypto::Poseidon2Vendor::kPlonky3, PackedF,
                            TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1,
                            TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_ALPHA>;

tachyon_sp1_baby_bear_poseidon2_two_adic_fri*
tachyon_sp1_baby_bear_poseidon2_two_adic_fri_create(uint32_t log_blowup,
                                                    size_t num_queries,
                                                    size_t proof_of_work_bits) {
  ExtF::Init();
  ExtPackedF::Init();

  math::Matrix<F> ark(TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_FULL_ROUNDS +
                          TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
                      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH);
  for (Eigen::Index r = 0; r < ark.rows(); ++r) {
    for (Eigen::Index c = 0; c < ark.cols(); ++c) {
      ark(r, c) = F(kRoundConstants[r][c] % F::Config::kModulus);
    }
  }

  math::Matrix<PackedF> packed_ark(
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_FULL_ROUNDS +
          TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH);
  for (Eigen::Index r = 0; r < ark.rows(); ++r) {
    for (Eigen::Index c = 0; c < ark.cols(); ++c) {
      packed_ark(r, c) = PackedF::Broadcast(ark(r, c));
    }
  }

  auto config = crypto::Poseidon2Config<Params>::Create(
      crypto::GetPoseidon2InternalShiftArray<Params>(), std::move(ark));
  Poseidon2 sponge(std::move(config));
  Hasher hasher(sponge);
  Compressor compressor(std::move(sponge));

  auto packed_config = crypto::Poseidon2Config<PackedParams>::Create(
      crypto::GetPoseidon2InternalShiftArray<PackedParams>(),
      std::move(packed_ark));
  PackedPoseidon2 packed_sponge(std::move(packed_config));
  PackedHasher packed_hasher(packed_sponge);
  PackedCompressor packed_compressor(std::move(packed_sponge));
  MMCS mmcs(hasher, packed_hasher, compressor, packed_compressor);

  ChallengeMMCS challenge_mmcs(
      ExtMMCS(std::move(hasher), std::move(packed_hasher),
              std::move(compressor), std::move(packed_compressor)));

  crypto::FRIConfig<ChallengeMMCS> fri_config{
      log_blowup, num_queries, proof_of_work_bits, std::move(challenge_mmcs)};

  return c::base::c_cast(new PCS(std::move(mmcs), std::move(fri_config)));
}

void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_destroy(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs) {
  delete c::base::native_cast(pcs);
}

void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_coset_lde_batch(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs,
    tachyon_baby_bear* values, size_t rows, size_t cols,
    tachyon_baby_bear* extended_values, tachyon_baby_bear shift) {
  PCS* native_pcs = c::base::native_cast(pcs);
  Eigen::Map<math::RowMajorMatrix<F>> matrix(c::base::native_cast(values),
                                             static_cast<Eigen::Index>(rows),
                                             static_cast<Eigen::Index>(cols));
  Eigen::Map<math::RowMajorMatrix<F>> extended_matrix(
      c::base::native_cast(extended_values),
      static_cast<Eigen::Index>(rows) << (native_pcs->config().log_blowup),
      static_cast<Eigen::Index>(cols));
  native_pcs->CosetLDEBatch(std::move(matrix), c::base::native_cast(shift),
                            extended_matrix);
}

void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_commit(
    tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs,
    tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec,
    tachyon_baby_bear* commitment,
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree** prover_data) {
  using Commitment = MMCS::Commitment;
  c::base::native_cast(pcs)->Commit(std::move(c::base::native_cast(*lde_vec)),
                                    reinterpret_cast<Commitment*>(commitment),
                                    reinterpret_cast<Tree**>(prover_data));
}

void tachyon_sp1_baby_bear_poseidon2_two_adic_fri_open(
    const tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs,
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec*
        prover_data_by_round,
    const tachyon_sp1_baby_bear_poseidon2_opening_points* points_by_round,
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger,
    tachyon_sp1_baby_bear_poseidon2_opened_values** opened_values,
    tachyon_sp1_baby_bear_poseidon2_fri_proof** proof) {
  c::base::native_cast(pcs)->CreateOpeningProof(
      c::base::native_cast(*prover_data_by_round),
      c::base::native_cast(*points_by_round), c::base::native_cast(*challenger),
      c::base::native_cast(
          reinterpret_cast<tachyon_sp1_baby_bear_poseidon2_opened_values*>(
              *opened_values)),
      c::base::native_cast(
          reinterpret_cast<tachyon_sp1_baby_bear_poseidon2_fri_proof*>(
              *proof)));
}

bool tachyon_sp1_baby_bear_poseidon2_two_adic_fri_verify(
    const tachyon_sp1_baby_bear_poseidon2_two_adic_fri* pcs,
    const tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitments_by_round,
    const tachyon_sp1_baby_bear_poseidon2_domains* domains_by_round,
    const tachyon_sp1_baby_bear_poseidon2_opening_points* points_by_round,
    const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values_by_round,
    const tachyon_sp1_baby_bear_poseidon2_fri_proof* proof,
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger) {
  return c::base::native_cast(pcs)->VerifyOpeningProof(
      c::base::native_cast(*commitments_by_round),
      c::base::native_cast(*domains_by_round),
      c::base::native_cast(*points_by_round),
      c::base::native_cast(*opened_values_by_round),
      c::base::native_cast(*proof), c::base::native_cast(*challenger));
}
