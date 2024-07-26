#include "tachyon/zk/base/blinder.h"

#include <memory_resource>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/random.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::zk {

namespace {

class FakeRandomFieldGenerator : public RandomFieldGeneratorBase<math::GF7> {
 public:
  explicit FakeRandomFieldGenerator(const std::vector<math::GF7>& values)
      : values_(values) {}

  // RandomFieldGeneratorBase<math::GF7> methods
  math::GF7 Generate() override {
    CHECK_LT(idx_, values_.size());
    return values_[idx_++];
  }

 private:
  const std::vector<math::GF7>& values_;
  size_t idx_ = 0;
};

}  // namespace

TEST(BlinderUnittest, Blind) {
  constexpr size_t kMaxDegree = 16;
  constexpr RowIndex kBlindingFactors = 10;

  using Evals = math::UnivariateEvaluations<math::GF7, kMaxDegree>;

  std::vector<math::GF7> blinding_values = base::CreateVector(
      kBlindingFactors + 1, []() { return math::GF7::Random(); });

  for (size_t i = 0; i < 2; ++i) {
    bool include_last_row = i == 0;

    FakeRandomFieldGenerator generator(blinding_values);
    Blinder<math::GF7> blinder(&generator, kBlindingFactors);

    RowIndex blinded_rows = kBlindingFactors;
    if (include_last_row) ++blinded_rows;
    std::pmr::vector<math::GF7> values = base::CreatePmrVector(
        blinded_rows - 1, []() { return math::GF7::Random(); });
    Evals evals(std::move(values));
    ASSERT_FALSE(blinder.Blind(evals, include_last_row));

    RowIndex rows = kBlindingFactors + 5;
    values = base::CreatePmrVector(rows, []() { return math::GF7::Random(); });
    evals = Evals(values);
    ASSERT_TRUE(blinder.Blind(evals, include_last_row));

    RowIndex not_blinded_rows = rows - blinded_rows;
    for (RowIndex i = 0; i < rows; ++i) {
      if (i < not_blinded_rows) {
        EXPECT_EQ(evals[i], values[i]);
      } else {
        EXPECT_EQ(evals[i], blinding_values[i - not_blinded_rows]);
      }
    }
  }
}

}  // namespace tachyon::zk
