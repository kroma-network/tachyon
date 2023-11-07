#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2_prover_test.h"

namespace tachyon::zk {
namespace {

class PermutationAssemblyTest : public Halo2ProverTest {
 public:
  void SetUp() override {
    Halo2ProverTest::SetUp();

    columns_ = {AnyColumn(0), AdviceColumn(1), FixedColumn(2),
                InstanceColumn(3)};
    argment_ = PermutationArgument(columns_);
    assembly_ = PermutationAssembly<PCS>::CreateForTesting(
        columns_, CycleStore(columns_.size(), kDomainSize));
  }

 protected:
  std::vector<AnyColumn> columns_;
  PermutationArgument argment_;
  PermutationAssembly<PCS> assembly_;
};

}  // namespace

TEST_F(PermutationAssemblyTest, GeneratePermutation) {
  // Check initial permutation polynomials w/o any copy.
  const Domain* domain = prover_->domain();
  std::vector<Evals> permutations = assembly_.GeneratePermutations(domain);

  LookupTable<PCS> lookup_table =
      LookupTable<PCS>::Construct(columns_.size(), domain);

  for (size_t i = 0; i < columns_.size(); ++i) {
    for (size_t j = 0; j <= kMaxDegree; ++j) {
      EXPECT_EQ(*permutations[i][j], lookup_table[Label(i, j)]);
    }
  }
}

// TODO(dongchangYoo): check if it produces the same value as zcash-halo2
TEST_F(PermutationAssemblyTest, BuildVerifyingKey) {}

// TODO(dongchangYoo): check if it produces the same value as zcash-halo2
TEST_F(PermutationAssemblyTest, BuildProvingKey) {}

}  // namespace tachyon::zk
