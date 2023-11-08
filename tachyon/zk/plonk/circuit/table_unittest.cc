#include "tachyon/zk/plonk/circuit/table.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::zk {

template <typename ColumnKeyType>
class TableTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = (size_t{1} << 3) - 1;
  constexpr static Phase kFirstPhase = Phase(0);

  using F = math::bn254::G1AffinePoint::ScalarField;
  using Evals = math::UnivariateEvaluations<F, kMaxDegree>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }

  void SetUp() override {
    for (size_t i = 0; i < 5; ++i) {
      fixed_columns_.push_back(
          math::UnivariateEvaluations<F, kMaxDegree>::Random());
      advice_columns_.push_back(
          math::UnivariateEvaluations<F, kMaxDegree>::Random());
      instance_columns_.push_back(
          math::UnivariateEvaluations<F, kMaxDegree>::Random());
    }
  }

 protected:
  std::vector<Evals> fixed_columns_;
  std::vector<Evals> advice_columns_;
  std::vector<Evals> instance_columns_;
};

using ColumnKeyTypes = testing::Types<ColumnKeyBase, AnyColumnKey>;
TYPED_TEST_SUITE(TableTest, ColumnKeyTypes);

TYPED_TEST(TableTest, GetColumns) {
  using ColumnKeyType = TypeParam;
  using Evals = typename TableTest<ColumnKeyType>::Evals;

  std::vector<ColumnKeyType> targets = {
      FixedColumnKey(1), AdviceColumnKey(1, kFirstPhase), InstanceColumnKey(1),
      FixedColumnKey(2), AdviceColumnKey(2, kFirstPhase), InstanceColumnKey(2),
  };

  Table<Evals> table(absl::MakeConstSpan(this->fixed_columns_),
                     absl::MakeConstSpan(this->advice_columns_),
                     absl::MakeConstSpan(this->instance_columns_));

  std::vector<Ref<const Evals>> evals_refs = table.GetColumns(targets);

  EXPECT_EQ(*evals_refs[0], this->fixed_columns_[1]);
  EXPECT_EQ(*evals_refs[1], this->advice_columns_[1]);
  EXPECT_EQ(*evals_refs[2], this->instance_columns_[1]);
  EXPECT_EQ(*evals_refs[3], this->fixed_columns_[2]);
  EXPECT_EQ(*evals_refs[4], this->advice_columns_[2]);
  EXPECT_EQ(*evals_refs[5], this->instance_columns_[2]);
}

}  // namespace tachyon::zk
