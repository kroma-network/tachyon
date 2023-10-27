#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::zk {
namespace {

constexpr static size_t MaxDegree = 7;

using Curve = math::bn254::G1Curve;
using F = Curve::ScalarField;
using Evals = PermutationAssembly<Curve, MaxDegree>::Evals;

class PermutationAssemblyTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Curve::Init(); }

  void SetUp() override {
    columns_ = {AnyColumn(0), AdviceColumn(1), FixedColumn(2),
                InstanceColumn(3)};
    argment_ = PermutationArgument(columns_);
    assembly_ = PermutationAssembly<Curve, MaxDegree>(argment_);

    domain_ = math::UnivariateEvaluationDomainFactory<F, MaxDegree>::Create(
        MaxDegree + 1);
  }

 protected:
  std::vector<AnyColumn> columns_;
  PermutationArgument argment_;
  PermutationAssembly<Curve, MaxDegree> assembly_;
  std::unique_ptr<math::UnivariateEvaluationDomain<F, MaxDegree>> domain_;
};

}  // namespace

TEST_F(PermutationAssemblyTest, GeneratePermutation) {
  // Check initial permutation polynomials w/o any copy.
  std::vector<Evals> permutations =
      assembly_.GeneratePermutations(domain_.get());

  LookupTable<F, MaxDegree> lookup_table =
      LookupTable<F, MaxDegree>::Construct(columns_.size(), domain_.get());

  for (size_t i = 0; i < columns_.size(); ++i) {
    for (size_t j = 0; j <= MaxDegree; ++j) {
      EXPECT_EQ(*permutations[i][j], lookup_table[Label(i, j)]);
    }
  }
}

// TODO(dongchangYoo): check if it produces the same value as zcash-halo2
TEST_F(PermutationAssemblyTest, BuildVerifyingKey) {}

// TODO(dongchangYoo): check if it produces the same value as zcash-halo2
TEST_F(PermutationAssemblyTest, BuildProvingKey) {}

}  // namespace tachyon::zk
