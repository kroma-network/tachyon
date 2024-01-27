#include "tachyon/zk/base/blinder.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/random.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk {

namespace {

class FakeEvals {
 public:
  FakeEvals() = default;
  explicit FakeEvals(const std::vector<math::GF7>& evaluations)
      : evaluations_(evaluations) {}

  const std::vector<math::GF7>& evaluations() const { return evaluations_; }
  std::vector<math::GF7>& evaluations() { return evaluations_; }

  math::GF7* operator[](size_t i) { return &evaluations_[i]; }
  const math::GF7* operator[](size_t i) const { return &evaluations_[i]; }

  size_t NumElements() const { return evaluations_.size(); }

 private:
  std::vector<math::GF7> evaluations_;
};

struct FakePCS {
  using Field = math::GF7;
  using Evals = FakeEvals;
};

class FakeRandomFieldGenerator : public RandomFieldGeneratorBase<math::GF7> {
 public:
  explicit FakeRandomFieldGenerator(const std::vector<math::GF7>& values)
      : values_(values) {}

  // RandomFieldGenerator<math::GF7> methods
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
  constexpr RowIndex kBlindingFactors = 10;
  std::vector<math::GF7> blinding_values = base::CreateVector(
      kBlindingFactors, []() { return math::GF7::Random(); });
  FakeRandomFieldGenerator generator(blinding_values);
  Blinder<FakePCS> blinder(&generator, kBlindingFactors);

  std::vector<math::GF7> evals = base::CreateVector(
      kBlindingFactors - 1, []() { return math::GF7::Random(); });

  FakeEvals fake_evals(evals);
  ASSERT_FALSE(blinder.Blind(fake_evals));

  evals = base::CreateVector(kBlindingFactors + 5,
                             []() { return math::GF7::Random(); });
  fake_evals = FakeEvals(evals);
  ASSERT_TRUE(blinder.Blind(fake_evals));

  for (size_t i = 0; i < 5; ++i) {
    if (i < 5) {
      EXPECT_EQ(evals[i], fake_evals.evaluations()[i]);
    } else {
      EXPECT_EQ(evals[i], blinding_values[i - 5]);
    }
  }
}

}  // namespace tachyon::zk
