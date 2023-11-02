#include "tachyon/zk/plonk/permutation/permutation_assembly.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::zk {
namespace {

class PermutationAssemblyTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = 7;
  constexpr static size_t kRows = kMaxDegree + 1;

  using PCS =
      crypto::KZGCommitmentScheme<math::bn254::G1AffinePoint,
                                  math::bn254::G2AffinePoint, kMaxDegree,
                                  math::bn254::G1AffinePoint>;
  using F = PCS::Field;
  using Evals = PCS::Evals;
  using Domain = PCS::Domain;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }

  void SetUp() override {
    columns_ = {AnyColumn(0), AdviceColumn(1), FixedColumn(2),
                InstanceColumn(3)};
    argment_ = PermutationArgument(columns_);
    assembly_ = PermutationAssembly<PCS>::CreateForTesting(
        columns_, CycleStore(columns_.size(), kRows));
    domain_ =
        math::UnivariateEvaluationDomainFactory<F, kMaxDegree>::Create(kRows);
  }

 protected:
  std::vector<AnyColumn> columns_;
  PermutationArgument argment_;
  PermutationAssembly<PCS> assembly_;
  std::unique_ptr<Domain> domain_;
};

}  // namespace

TEST_F(PermutationAssemblyTest, GeneratePermutation) {
  // Check initial permutation polynomials w/o any copy.
  std::vector<Evals> permutations =
      assembly_.GeneratePermutations(domain_.get());

  LookupTable<PCS> lookup_table =
      LookupTable<PCS>::Construct(columns_.size(), domain_.get());

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
