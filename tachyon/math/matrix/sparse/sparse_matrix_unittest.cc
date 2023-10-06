#include "tachyon/math/matrix/sparse/sparse_matrix.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::math {

class SparseMatrixTest : public testing::Test {
 public:
  void SetUp() override {
    std::vector<std::vector<Eigen::Triplet<GF7>>> coefficients_list;
    eigen_matrices_.push_back(Eigen::SparseMatrix<GF7>(4, 6));
    coefficients_list.push_back({
        {0, 3, GF7(1)},
        {1, 4, GF7(2)},
        {1, 5, GF7(3)},
        {2, 0, GF7(4)},
        {2, 1, GF7(5)},
        {3, 2, GF7(6)},
    });
    for (size_t i = 0; i < eigen_matrices_.size(); ++i) {
      eigen_matrices_[i].setFromTriplets(coefficients_list[i].begin(),
                                         coefficients_list[i].end());
      ell_matrices_.push_back(
          ELLSparseMatrix<GF7>::FromEigenSparseMatrix(eigen_matrices_[i]));
      csr_matrices_.push_back(
          CSRSparseMatrix<GF7>::FromEigenSparseMatrix(eigen_matrices_[i]));
    }
  }

 protected:
  std::vector<Eigen::SparseMatrix<GF7>> eigen_matrices_;
  std::vector<ELLSparseMatrix<GF7>> ell_matrices_;
  std::vector<CSRSparseMatrix<GF7>> csr_matrices_;
};

TEST_F(SparseMatrixTest, ELLSparseMatrixFromEigenMatrix) {
  for (size_t i = 0; i < eigen_matrices_.size(); ++i) {
    EXPECT_EQ(ELLSparseMatrix<GF7>::FromEigenSparseMatrix(eigen_matrices_[i]),
              ell_matrices_[i]);
    EXPECT_EQ(
        ELLSparseMatrix<GF7>::FromEigenSparseMatrix(
            Eigen::SparseMatrix<GF7, Eigen::RowMajor>(eigen_matrices_[i])),
        ell_matrices_[i]);
  }
}

TEST_F(SparseMatrixTest, ELLSparseMatrixSort) {
  // clang-format off
  ELLSparseMatrix<GF7> test_data({
    {{3, GF7(1)}             },
    {{5, GF7(3)}, {4, GF7(2)}},
    {{0, GF7(4)}, {1, GF7(5)}},
    {{2, GF7(6)}             },
  });
  // clang-format on
  EXPECT_FALSE(test_data.IsSorted());

  test_data.Sort();
  EXPECT_TRUE(test_data.IsSorted());
}

TEST_F(SparseMatrixTest, ELLSparseMatrixGetXXX) {
  // clang-format off
  struct {
    size_t max_cols;
    size_t non_zeros;
    std::vector<std::vector<GF7>> data;
    std::vector<std::vector<size_t>> column_indices;
  } answers[] = {
    {
      6,
      6,
      {
        {{GF7(1)}},
        {{GF7(2), GF7(3)}},
        {{GF7(4), GF7(5)}},
        {{GF7(6)}}
      },
      {
        {std::vector<size_t>{3}},
        {std::vector<size_t>{4, 5}},
        {std::vector<size_t>{0, 1}},
        {std::vector<size_t>{2}}
      },
    },
  };
  // clang-format on
  for (size_t i = 0; i < ell_matrices_.size(); ++i) {
    EXPECT_EQ(ell_matrices_[i].MaxCols(), answers[i].max_cols);
    EXPECT_EQ(ell_matrices_[i].NonZeros(), answers[i].non_zeros);
    EXPECT_THAT(ell_matrices_[i].GetData(),
                testing::ContainerEq(answers[i].data));
    EXPECT_THAT(ell_matrices_[i].GetColumnIndices(),
                testing::ContainerEq(answers[i].column_indices));
  }
}

TEST_F(SparseMatrixTest, ELLSparseMatrixToCSR) {
  for (size_t i = 0; i < ell_matrices_.size(); ++i) {
    EXPECT_EQ(ell_matrices_[i].ToCSR(), csr_matrices_[i]);
  }
}

TEST_F(SparseMatrixTest, ELLSparseMatrixToEigenSparseMatrix) {
  for (size_t i = 0; i < ell_matrices_.size(); ++i) {
    EXPECT_EQ(ELLSparseMatrix<GF7>::FromEigenSparseMatrix(
                  ell_matrices_[i].ToEigenSparseMatrix()),
              ell_matrices_[i]);
  }
}

TEST_F(SparseMatrixTest, CSRSparseMatrixFromEigenMatrix) {
  for (size_t i = 0; i < eigen_matrices_.size(); ++i) {
    EXPECT_EQ(CSRSparseMatrix<GF7>::FromEigenSparseMatrix(eigen_matrices_[i]),
              csr_matrices_[i]);
    EXPECT_EQ(
        CSRSparseMatrix<GF7>::FromEigenSparseMatrix(
            Eigen::SparseMatrix<GF7, Eigen::RowMajor>(eigen_matrices_[i])),
        csr_matrices_[i]);
  }
}

TEST_F(SparseMatrixTest, CSRSparseMatrixSort) {
  // clang-format off
  CSRSparseMatrix<GF7> test_data(
    {{3, GF7(1)}, {5, GF7(3)}, {4, GF7(2)}, {0, GF7(4)}, {1, GF7(5)}, {2, GF7(6)}},
    {0, 1, 3, 5, 6});
  // clang-format on
  EXPECT_FALSE(test_data.IsSorted());

  test_data.Sort();
  EXPECT_TRUE(test_data.IsSorted());
}

TEST_F(SparseMatrixTest, CSRSparseMatrixGetXXX) {
  // clang-format off
  struct {
    size_t max_cols;
    size_t non_zeros;
    std::vector<GF7> data;
    std::vector<size_t> column_indices;
  } answers[] = {
    {
      6,
      6,
      {GF7(1), GF7(2), GF7(3), GF7(4), GF7(5), GF7(6)},
      {3, 4, 5, 0, 1, 2},
    },
  };
  // clang-format on
  for (size_t i = 0; i < csr_matrices_.size(); ++i) {
    EXPECT_EQ(csr_matrices_[i].MaxCols(), answers[i].max_cols);
    EXPECT_EQ(csr_matrices_[i].NonZeros(), answers[i].non_zeros);
    EXPECT_THAT(csr_matrices_[i].GetData(),
                testing::ContainerEq(answers[i].data));
    EXPECT_THAT(csr_matrices_[i].GetColumnIndices(),
                testing::ContainerEq(answers[i].column_indices));
  }
}

TEST_F(SparseMatrixTest, CSRSparseMatrixToELL) {
  for (size_t i = 0; i < csr_matrices_.size(); ++i) {
    EXPECT_EQ(csr_matrices_[i].ToELL(), ell_matrices_[i]);
  }
}

TEST_F(SparseMatrixTest, CSRSparseMatrixToEigenSparseMatrix) {
  for (size_t i = 0; i < csr_matrices_.size(); ++i) {
    EXPECT_EQ(CSRSparseMatrix<GF7>::FromEigenSparseMatrix(
                  csr_matrices_[i].ToEigenSparseMatrix()),
              csr_matrices_[i]);
  }
}

}  // namespace tachyon::math
