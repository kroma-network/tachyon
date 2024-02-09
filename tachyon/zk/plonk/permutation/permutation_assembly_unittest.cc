#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk::plonk {
namespace {

class PermutationAssemblyTest : public halo2::ProverTest {
 public:
  void SetUp() override {
    halo2::ProverTest::SetUp();

    columns_ = {AnyColumnKey(0), AdviceColumnKey(1), FixedColumnKey(2),
                InstanceColumnKey(3)};
    argment_ = PermutationArgument(columns_);
    assembly_ = PermutationAssembly::CreateForTesting(
        columns_, CycleStore(columns_.size(), prover_->pcs().N()),
        prover_->pcs().N());
  }

 protected:
  std::vector<AnyColumnKey> columns_;
  PermutationArgument argment_;
  PermutationAssembly assembly_;
};

}  // namespace

TEST_F(PermutationAssemblyTest, GeneratePermutation) {
  // Check initial permutation polynomials w/o any copy.
  const Domain* domain = prover_->domain();
  std::vector<Evals> permutations =
      assembly_.GeneratePermutations<Evals>(domain);

  size_t n = prover_->pcs().N();
  UnpermutedTable<Evals> unpermuted_table =
      UnpermutedTable<Evals>::Construct(columns_.size(), n, domain);

  for (size_t i = 0; i < columns_.size(); ++i) {
    for (size_t j = 0; j < n; ++j) {
      EXPECT_EQ(permutations[i][j], unpermuted_table[Label(i, j)]);
    }
  }
}

TEST_F(PermutationAssemblyTest, BuildKeys) {
  const PCS& pcs = prover_->pcs();

  std::vector<Evals> permutations =
      assembly_.GeneratePermutations<Evals>(prover_->domain());
  size_t permutations_size = permutations.size();
  PermutationVerifyingKey<Commitment> vk =
      assembly_.BuildVerifyingKey(prover_.get(), permutations);
  EXPECT_EQ(permutations_size, vk.commitments().size());

  PermutationProvingKey<Poly, Evals> pk =
      assembly_.BuildProvingKey(prover_.get(), std::move(permutations));
  EXPECT_EQ(permutations_size, pk.polys().size());

  for (size_t i = 0; i < columns_.size(); ++i) {
    Commitment commitment_evals;
    ASSERT_TRUE(pcs.CommitLagrange(pk.permutations()[i], &commitment_evals));

    Commitment commitment_poly;
    ASSERT_TRUE(pcs.Commit(pk.polys()[i], &commitment_poly));

    // |polys| and |permutations| in the |PermutationProvingKey| represent
    // the same polynomial. so the commitments must be equal to each other.
    EXPECT_EQ(commitment_evals, commitment_poly);

    // |commitments| of |PermutationVerifyingKey| are commitments of
    // |permutations| of |PermutationProvingKey|.
    EXPECT_EQ(commitment_evals, vk.commitments()[i]);
  }
}

}  // namespace tachyon::zk::plonk
