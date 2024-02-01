#include "tachyon/zk/plonk/constraint_system/exclusion_matrix.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tachyon::zk::plonk {

namespace {

class ExclusionMatrixTest : public ::testing::Test {
 public:
  void SetUp() override {
    selector_activations_ = {
        // [1, 0, 0, 0, 0, 0, 0, 0, 1]
        {true, false, false, false, false, false, false, false, true},
        // [1, 0, 0, 0, 0, 0, 0, 1, 0]
        {true, false, false, false, false, false, false, true, false},
        // [1, 0, 0, 0, 0, 0, 1, 0, 0]
        {true, false, false, false, false, false, true, false, false},
        // [0, 1, 0, 0, 0, 1, 1, 1, 0]
        {false, true, false, false, false, true, true, true, false},
        // [0, 1, 0, 0, 1, 0, 1, 0, 1]
        {false, true, false, false, true, false, true, false, true},
        // [0, 1, 0, 1, 0, 0, 0, 1, 1]
        {false, true, false, true, false, false, false, true, true},
        // [0, 0, 1, 1, 1, 0, 0, 0, 0]
        {false, false, true, true, true, false, false, false, false},
        // [0, 0, 1, 1, 0, 1, 0, 0, 0]
        {false, false, true, true, false, true, false, false, false},
        // [0, 0, 1, 0, 1, 1, 0, 0, 0]
        {false, false, true, false, true, true, false, false, false}};

    // |exclusion_matrix| should be the lower triangular of the whole matrix.
    // +-----+
    // |  1  |
    // +-----+-----+
    // |  1  |  1  |
    // +-----+-----+-----+
    // |  0  |  1  |  1  |
    // +-----+-----+-----+-----+
    // |  1  |  0  |  1  |  1  |
    // +-----+-----+-----+-----+-----+
    // |  1  |  1  |  0  |  1  |  1  |
    // +-----+-----+-----+-----+-----+-----+
    // |  0  |  0  |  0  |  0  |  1  |  1  |
    // +-----+-----+-----+-----+-----+-----+-----+
    // |  0  |  0  |  0  |  1  |  0  |  1  |  1  |
    // +-----+-----+-----+-----+-----+-----+-----+-----+
    // |  0  |  0  |  0  |  1  |  1  |  0  |  1  |  1  |
    // +-----+-----+-----+-----+-----+-----+-----+-----+
    expected_matrix_ = {{},
                        {true},
                        {true, true},
                        {false, true, true},
                        {true, false, true, true},
                        {true, true, false, true, true},
                        {false, false, false, false, true, true},
                        {false, false, false, true, false, true, true},
                        {false, false, false, true, true, false, true, true}};

    selectors_ =
        base::CreateVector(selector_activations_.size(), [this](size_t i) {
          return SelectorDescription(i, &selector_activations_[i], 1);
        });
  }

 protected:
  std::vector<std::vector<bool>> selector_activations_;
  std::vector<std::vector<bool>> expected_matrix_;
  std::vector<SelectorDescription> selectors_;
};

}  // namespace

TEST_F(ExclusionMatrixTest, ConstructExclusionMatrix) {
  ExclusionMatrix exclusion_matrix(selectors_);

  std::vector<std::vector<bool>> actual_matrix =
      exclusion_matrix.lower_triangular_matrix();
  EXPECT_EQ(actual_matrix, expected_matrix_);
}

TEST_F(ExclusionMatrixTest, IsExclusive) {
  ExclusionMatrix exclusion_matrix(selectors_);

  // IsExclusive(i, j) returns the value of the exclusion matrix at (i, j).
  for (size_t i = 0; i < selectors_.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      EXPECT_EQ(exclusion_matrix.IsExclusive(i, j), expected_matrix_[i][j]);
    }
  }

  // Since the matrix is symmetric, the lower triangular and upper triangular
  // matrix must be the same.
  for (size_t i = 0; i < selectors_.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      EXPECT_EQ(exclusion_matrix.IsExclusive(i, j),
                exclusion_matrix.IsExclusive(j, i));
    }
  }
}

}  // namespace tachyon::zk::plonk
