#include "tachyon/crypto/commitments/fri/two_adic_fri.h"

#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/crypto/challenger/duplex_challenger.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/extension_field_merkle_tree_mmcs.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree_mmcs.h"
#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

namespace {

constexpr size_t kRate = 8;
constexpr size_t kChunk = 8;
constexpr size_t kN = 2;

using F = math::BabyBear;
using ExtF = math::BabyBear4;
using PackedF = math::PackedBabyBear;
using ExtPackedF = math::PackedBabyBear4;
using Params = Poseidon2Params<Poseidon2Vendor::kPlonky3,
                               Poseidon2Vendor::kPlonky3, F, 15, 7>;
using PackedParams = Poseidon2Params<Poseidon2Vendor::kPlonky3,
                                     Poseidon2Vendor::kPlonky3, PackedF, 15, 7>;
using Domain = TwoAdicMultiplicativeCoset<F>;
using Poseidon2 = Poseidon2Sponge<Params>;
using PackedPoseidon2 = Poseidon2Sponge<PackedParams>;
using MyHasher = PaddingFreeSponge<Poseidon2, kRate, kChunk>;
using MyPackedHasher = PaddingFreeSponge<PackedPoseidon2, kRate, kChunk>;
using MyCompressor = TruncatedPermutation<Poseidon2, kChunk, kN>;
using MyPackedCompressor = TruncatedPermutation<PackedPoseidon2, kChunk, kN>;
using MMCS = FieldMerkleTreeMMCS<F, MyHasher, MyPackedHasher, MyCompressor,
                                 MyPackedCompressor, kChunk>;
using ExtMMCS = FieldMerkleTreeMMCS<ExtF, MyHasher, MyPackedHasher,
                                    MyCompressor, MyPackedCompressor, kChunk>;
using ChallengeMMCS = ExtensionFieldMerkleTreeMMCS<ExtF, ExtMMCS>;
using Challenger = DuplexChallenger<Poseidon2, kRate>;
using Coset = TwoAdicMultiplicativeCoset<F>;
using MyPCS = TwoAdicFRI<ExtF, MMCS, ChallengeMMCS, Challenger>;

class TwoAdicFRITest : public testing::Test {
 public:
  TwoAdicFRITest() = default;

  static void SetUpTestSuite() {
    ExtF::Init();
    ExtPackedF::Init();
  }

  void SetUp() override {
    auto config = Poseidon2Config<Params>::Create(
        GetPoseidon2InternalShiftArray<Params>());
    Poseidon2 sponge(std::move(config));
    MyHasher hasher(sponge);
    MyCompressor compressor(sponge);

    auto packed_config = Poseidon2Config<PackedParams>::Create(
        GetPoseidon2InternalShiftArray<PackedParams>());
    PackedPoseidon2 packed_sponge(std::move(packed_config));
    MyPackedHasher packed_hasher(packed_sponge);
    MyPackedCompressor packed_compressor(std::move(packed_sponge));
    MMCS mmcs(hasher, packed_hasher, compressor, packed_compressor);

    ChallengeMMCS challenge_mmcs(
        ExtMMCS(std::move(hasher), std::move(packed_hasher),
                std::move(compressor), std::move(packed_compressor)));

    // TODO(ashjeong): Include separate test for |log_blowup| = 2
    FRIConfig<ChallengeMMCS> fri_config{1, 10, 8, std::move(challenge_mmcs)};

    pcs_ = MyPCS(std::move(mmcs), std::move(fri_config));
    challenger_ = Challenger(std::move(sponge));
  }

  void TestProtocol(std::vector<std::vector<uint32_t>> log_degrees_by_round) {
    using Commitment = typename MMCS::Commitment;
    using ProverData = typename MMCS::ProverData;
    using OpenedValues =
        std::vector<std::vector<std::vector<std::vector<ExtF>>>>;
    using Proof = FRIProof<MyPCS>;

    size_t num_rounds = log_degrees_by_round.size();
    std::vector<std::vector<Domain>> domains_by_round(num_rounds);
    std::vector<Commitment> commits_by_round(num_rounds);
    std::vector<std::unique_ptr<ProverData>> data_by_round(num_rounds);
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
      data_by_round[round].reset(new ProverData());
      ASSERT_TRUE(pcs_.Commit(inner_domains, inner_polys,
                              &commits_by_round[round],
                              data_by_round[round].get()));
      domains_by_round[round] = std::move(inner_domains);
    }
    p_challenger.ObserveContainer2D(commits_by_round);
    ExtF zeta = p_challenger.template SampleExtElement<ExtF>();

    std::vector<std::vector<std::vector<ExtF>>> points_by_round(num_rounds);
    for (size_t round = 0; round < num_rounds; ++round) {
      points_by_round[round] = std::vector<std::vector<ExtF>>(
          log_degrees_by_round[round].size(), {zeta});
    }
    OpenedValues opened_values_by_round;
    Proof proof;
    ASSERT_TRUE(pcs_.CreateOpeningProof(data_by_round, points_by_round,
                                        p_challenger, &opened_values_by_round,
                                        &proof));
    ASSERT_EQ(opened_values_by_round.size(), num_rounds);

    // Verify the proof
    Challenger v_challenger = challenger_;
    v_challenger.ObserveContainer2D(commits_by_round);
    ExtF verifier_zeta = v_challenger.template SampleExtElement<ExtF>();
    ASSERT_EQ(verifier_zeta, zeta);

    ASSERT_TRUE(pcs_.VerifyOpeningProof(commits_by_round, domains_by_round,
                                        points_by_round, opened_values_by_round,
                                        proof, v_challenger));
  }

 protected:
  MyPCS pcs_;
  Challenger challenger_;
};

}  // namespace

TEST_F(TwoAdicFRITest, Single) {
  for (uint32_t i = 3; i < 6; ++i) TestProtocol({{i}});
}

TEST_F(TwoAdicFRITest, ManyEqual) {
  for (uint32_t i = 2; i < 5; ++i) TestProtocol({std::vector<uint32_t>(5, i)});
}

TEST_F(TwoAdicFRITest, ManyDifferent) {
  for (uint32_t i = 2; i < 4; ++i) {
    std::vector<uint32_t> input(i);
    for (uint32_t j = 3; j < 3 + i; ++j) {
      input[j - 3] = j;
    }
    TestProtocol({input});
  }
}

TEST_F(TwoAdicFRITest, ManyDifferentRev) {
  for (uint32_t i = 2; i < 4; ++i) {
    std::vector<uint32_t> input(i);
    for (uint32_t j = 3 + i - 1; j >= 3; --j) {
      input[j - 3] = j;
    }
    TestProtocol({input});
  }
}

TEST_F(TwoAdicFRITest, MultipleRounds) {
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
