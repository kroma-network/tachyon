#include "tachyon/zk/plonk/constraint_system/selector_compressor.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/functional/callback.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/zk/plonk/expressions/expression_factory.h"

namespace tachyon::zk::plonk {

namespace {

using F = math::GF7;
using AllocateFixedColumnCallback =
    base::RepeatingCallback<std::unique_ptr<Expression<F>>()>;

class SelectorCompressorTest : public math::FiniteFieldTest<F> {
 public:
  void SetUp() override {
    // [1, 0, 0, 0, 0, 0, 0, 0, 1],
    selectors_in_.push_back(
        {true, false, false, false, false, false, false, false, true});
    // [1, 0, 0, 0, 0, 0, 0, 1, 0],
    selectors_in_.push_back(
        {true, false, false, false, false, false, false, true, false});
    // [1, 0, 0, 0, 0, 0, 1, 0, 0],
    selectors_in_.push_back(
        {true, false, false, false, false, false, true, false, false});
    // [0, 1, 0, 0, 0, 1, 1, 1, 0],
    selectors_in_.push_back(
        {false, true, false, false, false, true, true, true, false});
    // [0, 1, 0, 0, 1, 0, 1, 0, 1],
    selectors_in_.push_back(
        {false, true, false, false, true, false, true, false, true});
    // [0, 1, 0, 1, 0, 0, 0, 1, 1],
    selectors_in_.push_back(
        {false, true, false, true, false, false, false, true, true});
    // [0, 0, 1, 1, 1, 0, 0, 0, 0],
    selectors_in_.push_back(
        {false, false, true, true, true, false, false, false, false});
    // [0, 0, 1, 1, 0, 1, 0, 0, 0],
    selectors_in_.push_back(
        {false, false, true, true, false, true, false, false, false});
    // [0, 0, 1, 0, 1, 1, 0, 0, 0],
    selectors_in_.push_back(
        {false, false, true, false, true, true, false, false, false});
    // [1, 1, 1, 1, 1, 1, 1, 1, 1],
    selectors_in_.push_back(
        {true, true, true, true, true, true, true, true, true});

    degrees_ = {3, 3, 3, 3, 3, 3, 7, 7, 7, 0};

    callback_ = [this]() {
      auto ret = ExpressionFactory<F>::Fixed(
          FixedQuery(new_column_index_, Rotation::Cur(),
                     FixedColumnKey(new_column_index_)));
      new_column_index_++;
      return ret;
    };
  }

 protected:
  size_t new_column_index_ = 0;
  std::vector<std::vector<bool>> selectors_in_;
  std::vector<size_t> degrees_;
  AllocateFixedColumnCallback callback_;
};

}  // namespace

TEST_F(SelectorCompressorTest, HandleZeroDegreeSelectors) {
  SelectorCompressor<F> selector_compressor;
  selector_compressor.callback_ = callback_;

  selectors_in_.push_back(
      {false, true, false, true, false, true, false, true, false});
  selectors_in_.push_back(
      {false, false, true, true, true, false, false, false, false});
  degrees_.push_back(0);
  degrees_.push_back(0);

  selector_compressor.selectors_ = base::Map(
      selectors_in_, [this](size_t i, const std::vector<bool>& activations) {
        size_t max_degree = degrees_[i];
        return SelectorDescription(i, &activations, max_degree);
      });

  // All selectors of degree 0 should be filtered out.
  selector_compressor.HandleZeroDegreeSelectors();
  EXPECT_EQ(selector_compressor.selectors_.size(), 9);
  for (const SelectorDescription& selector : selector_compressor.selectors_) {
    EXPECT_NE(selector.max_degree(), 0);
  }
}

TEST_F(SelectorCompressorTest, ConstructCombinedSelector) {
  SelectorCompressor<F> selector_compressor;
  selector_compressor.callback_ = callback_;

  // s₀: SelectorDescription(0, [1, 0, 0, 0, 0, 0, 0, 0, 1], 3)
  // s₃: SelectorDescription(3, [0, 1, 0, 0, 0, 1, 1, 1, 0], 3)
  // s₆: SelectorDescription(6, [0, 0, 1, 1, 1, 0, 0, 0, 0], 7)
  // => [1, 2, 3, 3, 3, 2, 2, 2, 1]
  std::vector<SelectorDescription> before_combination = {
      {0, &selectors_in_[0], 3},
      {3, &selectors_in_[3], 3},
      {6, &selectors_in_[6], 7},
  };
  selector_compressor.ConstructCombinedSelector(9, before_combination);
  std::vector<F, base::memory::ReusingAllocator<F>>
      expected_combination_assignment = {F(1), F(2), F(3), F(3), F(3),
                                         F(2), F(2), F(2), F(1)};
  EXPECT_EQ(selector_compressor.combination_assignments()[0],
            expected_combination_assignment);
}

TEST_F(SelectorCompressorTest, Process) {
  std::vector<std::vector<F, base::memory::ReusingAllocator<F>>>
      expected_polys = {
          {F(1), F(1), F(1), F(1), F(1), F(1), F(1), F(1), F(1)},
          {F(1), F(2), F(3), F(3), F(3), F(2), F(2), F(2), F(1)},
          {F(1), F(2), F(3), F(3), F(2), F(3), F(2), F(1), F(2)},
          {F(1), F(2), F(3), F(2), F(3), F(3), F(1), F(2), F(2)},
      };

  std::vector<size_t> expected_selector_indices = {9, 0, 3, 6, 1,
                                                   4, 7, 2, 5, 8};

  SelectorCompressor<F> selector_compressor;

  selector_compressor.Process(selectors_in_, degrees_, 10, callback_);

  std::vector<size_t> actual_selector_indices =
      base::Map(selector_compressor.selector_assignments(),
                [](const SelectorAssignment<F>& assignment) {
                  return assignment.selector_index();
                });

  EXPECT_EQ(selector_compressor.combination_assignments(), expected_polys);
  EXPECT_EQ(actual_selector_indices, expected_selector_indices);
}

}  // namespace tachyon::zk::plonk
