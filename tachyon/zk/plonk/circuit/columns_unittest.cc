#include "tachyon/zk/plonk/circuit/columns.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::zk {

class ColumnsTest : public testing::Test {
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

TEST_F(ColumnsTest, FindColumns) {
  std::vector<ColumnData> targets = {
      FixedColumn(1), AdviceColumn(1, kFirstPhase), InstanceColumn(1),
      FixedColumn(2), AdviceColumn(2, kFirstPhase), InstanceColumn(2),
  };

  Columns<Evals> columns(absl::MakeConstSpan(fixed_columns_),
                         absl::MakeConstSpan(advice_columns_),
                         absl::MakeConstSpan(instance_columns_));

  std::vector<Ref<const Evals>> evals_refs = columns.GetColumns(targets);

  EXPECT_EQ(*evals_refs[0], fixed_columns_[1]);
  EXPECT_EQ(*evals_refs[1], advice_columns_[1]);
  EXPECT_EQ(*evals_refs[2], instance_columns_[1]);
  EXPECT_EQ(*evals_refs[3], fixed_columns_[2]);
  EXPECT_EQ(*evals_refs[4], advice_columns_[2]);
  EXPECT_EQ(*evals_refs[5], instance_columns_[2]);
}

}  // namespace tachyon::zk
