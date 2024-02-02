#include "tachyon/zk/plonk/base/ref_table.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::zk::plonk {

template <typename ColumnKey>
class RefTableTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = (size_t{1} << 3) - 1;

  using F = math::bn254::G1AffinePoint::ScalarField;
  using Evals = math::UnivariateEvaluations<F, kMaxDegree>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }

  void SetUp() override {
    for (size_t i = 0; i < 5; ++i) {
      fixed_columns_.push_back(
          math::UnivariateEvaluations<F, kMaxDegree>::Random(kMaxDegree));
      advice_columns_.push_back(
          math::UnivariateEvaluations<F, kMaxDegree>::Random(kMaxDegree));
      instance_columns_.push_back(
          math::UnivariateEvaluations<F, kMaxDegree>::Random(kMaxDegree));
    }
  }

 protected:
  std::vector<Evals> fixed_columns_;
  std::vector<Evals> advice_columns_;
  std::vector<Evals> instance_columns_;
};

using ColumnKeyTypes = testing::Types<ColumnKeyBase, AnyColumnKey>;
TYPED_TEST_SUITE(RefTableTest, ColumnKeyTypes);

TYPED_TEST(RefTableTest, GetColumns) {
  using ColumnKey = TypeParam;
  using Evals = typename RefTableTest<ColumnKey>::Evals;

  std::vector<ColumnKey> targets = {
      FixedColumnKey(1), AdviceColumnKey(1, kFirstPhase), InstanceColumnKey(1),
      FixedColumnKey(2), AdviceColumnKey(2, kFirstPhase), InstanceColumnKey(2),
  };

  RefTable<Evals> table(absl::MakeConstSpan(this->fixed_columns_),
                        absl::MakeConstSpan(this->advice_columns_),
                        absl::MakeConstSpan(this->instance_columns_));

  std::vector<base::Ref<const Evals>> evals_refs = table.GetColumns(targets);

  EXPECT_EQ(*evals_refs[0], this->fixed_columns_[1]);
  EXPECT_EQ(*evals_refs[1], this->advice_columns_[1]);
  EXPECT_EQ(*evals_refs[2], this->instance_columns_[1]);
  EXPECT_EQ(*evals_refs[3], this->fixed_columns_[2]);
  EXPECT_EQ(*evals_refs[4], this->advice_columns_[2]);
  EXPECT_EQ(*evals_refs[5], this->instance_columns_[2]);
}

}  // namespace tachyon::zk::plonk
