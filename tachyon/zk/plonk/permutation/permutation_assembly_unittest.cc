#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2/halo2_prover_test.h"

namespace tachyon::zk {
namespace {

class PermutationAssemblyTest : public Halo2ProverTest {
 public:
  void SetUp() override {
    Halo2ProverTest::SetUp();

    columns_ = {AnyColumnKey(0), AdviceColumnKey(1), FixedColumnKey(2),
                InstanceColumnKey(3)};
    argment_ = PermutationArgument(columns_);
    assembly_ = PermutationAssembly<PCS>::CreateForTesting(
        columns_, CycleStore(columns_.size(), prover_->pcs().N()),
        prover_->pcs().N());
  }

 protected:
  std::vector<AnyColumnKey> columns_;
  PermutationArgument argment_;
  PermutationAssembly<PCS> assembly_;
};

}  // namespace

TEST_F(PermutationAssemblyTest, GeneratePermutation) {
  // Check initial permutation polynomials w/o any copy.
  const Domain* domain = prover_->domain();
  std::vector<Evals> permutations = assembly_.GeneratePermutations(domain);

  size_t n = prover_->pcs().N();
  UnpermutedTable<Evals> unpermuted_table =
      UnpermutedTable<Evals>::Construct(columns_.size(), n, domain);

  for (size_t i = 0; i < columns_.size(); ++i) {
    for (size_t j = 0; j < n; ++j) {
      EXPECT_EQ(*permutations[i][j], unpermuted_table[Label(i, j)]);
    }
  }
}

TEST_F(PermutationAssemblyTest, BuildKeys) {
  const PCS& pcs = prover_->pcs();

  std::vector<Evals> permutations =
      assembly_.GeneratePermutations(prover_->domain());
  PermutationProvingKey<Poly, Evals> pk =
      assembly_.BuildProvingKey(prover_.get(), permutations);
  EXPECT_EQ(pk.permutations().size(), pk.polys().size());

  PermutationVerifyingKey<PCS> vk =
      assembly_.BuildVerifyingKey(prover_.get(), permutations);
  EXPECT_EQ(pk.permutations().size(), vk.commitments().size());

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

}  // namespace tachyon::zk
