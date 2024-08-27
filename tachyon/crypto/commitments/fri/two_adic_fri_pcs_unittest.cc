#include "tachyon/crypto/commitments/fri/two_adic_fri_pcs.h"

#include <tuple>

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/extension_field_merkle_tree_mmcs.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree_mmcs.h"
#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/zk/air/plonky3/base/two_adic_multiplicative_coset.h"
#include "tachyon/zk/air/plonky3/challenger/duplex_challenger.h"

namespace tachyon::crypto {

namespace {

constexpr size_t kRate = 8;
constexpr size_t kChunk = 8;
constexpr size_t kN = 2;

using F = math::BabyBear;
using ExtF = math::BabyBear4;
using PackedF = math::PackedBabyBear;
using ExtPackedF = math::PackedBabyBear4;
using Domain = zk::air::plonky3::TwoAdicMultiplicativeCoset<F>;
using Poseidon2 =
    Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<F>>>;
using PackedPoseidon2 = Poseidon2Sponge<
    Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<PackedF>>>;
using MyHasher = PaddingFreeSponge<Poseidon2, kRate, kChunk>;
using MyPackedHasher = PaddingFreeSponge<PackedPoseidon2, kRate, kChunk>;
using MyCompressor = TruncatedPermutation<Poseidon2, kChunk, kN>;
using MyPackedCompressor = TruncatedPermutation<PackedPoseidon2, kChunk, kN>;
using MMCS = FieldMerkleTreeMMCS<F, MyHasher, MyPackedHasher, MyCompressor,
                                 MyPackedCompressor, kChunk>;
using ExtMMCS = FieldMerkleTreeMMCS<ExtF, MyHasher, MyPackedHasher,
                                    MyCompressor, MyPackedCompressor, kChunk>;
using ChallengeMMCS = ExtensionFieldMerkleTreeMMCS<ExtF, ExtMMCS>;
using Challenger = zk::air::plonky3::DuplexChallenger<Poseidon2, 16, kRate>;
using Coset = zk::air::plonky3::TwoAdicMultiplicativeCoset<F>;
using MyPcs = TwoAdicFriPCS<ExtF, MMCS, ChallengeMMCS, Challenger, Coset>;

class TwoAdicFriPCSTest : public testing::Test {
 public:
  TwoAdicFriPCSTest() = default;

  static void SetUpTestSuite() {
    ExtF::Init();
    ExtPackedF::Init();
  }

  void SetUp() override {
    Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
        15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    Poseidon2 sponge(config);
    MyHasher hasher(sponge);
    MyCompressor compressor(sponge);

    Poseidon2Config<PackedF> packed_config =
        Poseidon2Config<PackedF>::CreateCustom(
            15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    PackedPoseidon2 packed_sponge(packed_config);
    MyPackedHasher packed_hasher(packed_sponge);
    MyPackedCompressor packed_compressor(std::move(packed_sponge));
    MMCS mmcs(hasher, packed_hasher, compressor, packed_compressor);

    ChallengeMMCS challenge_mmcs(
        ExtMMCS(std::move(hasher), std::move(packed_hasher),
                std::move(compressor), std::move(packed_compressor)));

    // TODO(ashjeong): Include separate test for |log_blowup| = 2
    TwoAdicFriConfig<ChallengeMMCS> fri_config{1, 10, 8,
                                               std::move(challenge_mmcs)};

    pcs_ = MyPcs(std::move(mmcs), std::move(fri_config));
    challenger_ = Challenger(std::move(sponge));
  }

  void TestProtocol(std::vector<std::vector<uint32_t>> log_degrees_by_round) {
    using Commitment = typename MMCS::Commitment;
    using ProverData = typename MMCS::ProverData;
    using OpenedValues =
        std::vector<std::vector<std::vector<std::vector<ExtF>>>>;
    using Proof =
        TwoAdicFriProof<ChallengeMMCS, std::vector<BatchOpening<MMCS>>, F>;
    using Claims = std::vector<std::tuple<ExtF, std::vector<ExtF>>>;

    size_t num_rounds = log_degrees_by_round.size();
    std::vector<std::vector<Domain>> domains_by_round(num_rounds);
    std::vector<Commitment> commits_by_round(num_rounds);
    std::vector<ProverData> data_by_round(num_rounds);
    Challenger p_challenger = challenger_;
    for (size_t round = 0; round < num_rounds; ++round) {
      const std::vector<uint32_t>& log_degrees = log_degrees_by_round[round];
      std::vector<Domain> inner_domains(log_degrees.size());
      std::vector<math::RowMajorMatrix<F>> inner_polys(log_degrees.size());
      for (size_t i = 0; i < log_degrees.size(); ++i) {
        size_t rows = size_t{1} << log_degrees[i];
        // TODO(ashjeong): make the latter number randomized from 0-10
        size_t cols = 5;
        inner_domains[i] = pcs_.GetNaturalDomainForDegree(rows);
        inner_polys[i] = math::RowMajorMatrix<F>::Random(rows, cols);
      }
      ASSERT_TRUE(pcs_.Commit(inner_domains, inner_polys,
                              &commits_by_round[round], &data_by_round[round]));
      domains_by_round[round] = std::move(inner_domains);
    }
    p_challenger.ObserveContainer2D(commits_by_round);
    ExtF zeta = p_challenger.template SampleExtElement<ExtF>();

    std::vector<std::vector<std::vector<ExtF>>> points_by_round(num_rounds);
    for (size_t round = 0; round < num_rounds; ++round) {
      points_by_round[round] = std::vector<std::vector<ExtF>>(
          log_degrees_by_round[round].size(), {zeta});
    }
    OpenedValues openings;
    Proof proof;
    ASSERT_TRUE(pcs_.CreateOpeningProof(data_by_round, points_by_round,
                                        p_challenger, &openings, &proof));
    ASSERT_EQ(openings.size(), num_rounds);

    // Verify the proof
    Challenger v_challenger = challenger_;
    v_challenger.ObserveContainer2D(commits_by_round);
    ExtF verifier_zeta = v_challenger.template SampleExtElement<ExtF>();
    ASSERT_EQ(verifier_zeta, zeta);

    std::vector<std::vector<Claims>> claims_by_round = base::CreateVector(
        num_rounds, [&domains_by_round, &zeta, &openings](size_t round) {
          return base::CreateVector(
              domains_by_round[round].size(),
              [round, &zeta, &openings](size_t i) {
                return Claims{std::make_tuple(zeta, openings[round][i][0])};
              });
        });
    ASSERT_TRUE(pcs_.VerifyOpeningProof(commits_by_round, domains_by_round,
                                        claims_by_round, proof, v_challenger));
  }

 protected:
  MyPcs pcs_;
  Challenger challenger_;
};

}  // namespace

TEST_F(TwoAdicFriPCSTest, Single) {
  for (uint32_t i = 3; i < 6; ++i) TestProtocol({{i}});
}

TEST_F(TwoAdicFriPCSTest, ManyEqual) {
  for (uint32_t i = 2; i < 5; ++i) TestProtocol({std::vector<uint32_t>(5, i)});
}

TEST_F(TwoAdicFriPCSTest, ManyDifferent) {
  for (uint32_t i = 2; i < 4; ++i) {
    std::vector<uint32_t> input(i);
    for (uint32_t j = 3; j < 3 + i; ++j) {
      input[j - 3] = j;
    }
    TestProtocol({input});
  }
}

TEST_F(TwoAdicFriPCSTest, ManyDifferentRev) {
  for (uint32_t i = 2; i < 4; ++i) {
    std::vector<uint32_t> input(i);
    for (uint32_t j = 3 + i - 1; j >= 3; --j) {
      input[j - 3] = j;
    }
    TestProtocol({input});
  }
}

TEST_F(TwoAdicFriPCSTest, MultipleRounds) {
  TestProtocol({{3}});
  TestProtocol({{3}, {3}});
  TestProtocol({{3}, {2}});
  TestProtocol({{2}, {3}});
  TestProtocol({{1, 2}});
  TestProtocol({{4, 2}, {4, 2}});
  TestProtocol({{2, 3}, {3, 3}});
  TestProtocol({{3, 3}, {2, 2}});
  TestProtocol({{2}, {3, 3}});
}

}  // namespace tachyon::crypto
