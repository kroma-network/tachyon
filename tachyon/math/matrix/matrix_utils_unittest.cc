#include "tachyon/math/matrix/matrix_utils.h"

#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::math {

class MatrixUtilsTest : public FiniteFieldTest<GF7> {};

TEST_F(MatrixUtilsTest, Circulant) {
  Matrix<GF7> circulant = MakeCirculant(Vector<GF7>{{GF7(2), GF7(3), GF7(4)}});
  Matrix<GF7> expected{{
      {GF7(2), GF7(4), GF7(3)},
      {GF7(3), GF7(2), GF7(4)},
      {GF7(4), GF7(3), GF7(2)},
  }};
  EXPECT_EQ(circulant, expected);
}

class MatrixPackingTest : public FiniteFieldTest<PackedBabyBear> {};

TEST_F(MatrixPackingTest, PackRowHorizontally) {
  constexpr size_t N = PackedBabyBear::N;
  constexpr size_t R = 3;

  {
    RowMajorMatrix<BabyBear> matrix =
        RowMajorMatrix<BabyBear>::Random(2 * N, 2 * N);
    Eigen::Block<RowMajorMatrix<BabyBear>> mat =
        matrix.block(0, 0, matrix.rows(), matrix.cols());
    std::vector<BabyBear*> remaining_values;
    std::vector<PackedBabyBear*> packed_values =
        PackRowHorizontally<PackedBabyBear>(mat, R, remaining_values);
    ASSERT_TRUE(remaining_values.empty());
    ASSERT_EQ(packed_values.size(), 2);
    for (size_t i = 0; i < packed_values.size(); ++i) {
      for (size_t j = 0; j < N; ++j) {
        EXPECT_EQ((*packed_values[i])[j], matrix(R, i * N + j));
      }
    }
  }
  {
    RowMajorMatrix<BabyBear> matrix =
        RowMajorMatrix<BabyBear>::Random(2 * N - 1, 2 * N - 1);
    Eigen::Block<RowMajorMatrix<BabyBear>> mat =
        matrix.block(0, 0, matrix.rows(), matrix.cols());
    std::vector<BabyBear*> remaining_values;
    std::vector<PackedBabyBear*> packed_values =
        PackRowHorizontally<PackedBabyBear>(mat, R, remaining_values);
    ASSERT_EQ(remaining_values.size(), N - 1);
    ASSERT_EQ(packed_values.size(), 1);
    for (size_t i = 0; i < remaining_values.size(); ++i) {
      EXPECT_EQ(*remaining_values[i], matrix(R, packed_values.size() * N + i));
    }
    for (size_t i = 0; i < packed_values.size(); ++i) {
      for (size_t j = 0; j < N; ++j) {
        EXPECT_EQ((*packed_values[i])[j], matrix(R, i * N + j));
      }
    }
  }
}

TEST_F(MatrixPackingTest, PackRowVerticallyWithPrimeField) {
  constexpr size_t N = PackedBabyBear::N;
  constexpr size_t R = 3;

  Matrix<BabyBear> matrix = Matrix<BabyBear>::Random(N, N);
  std::vector<PackedBabyBear> packed_values =
      PackRowVertically<PackedBabyBear>(matrix, R);
  ASSERT_EQ(packed_values.size(), N);
  for (size_t i = 0; i < packed_values.size(); ++i) {
    for (size_t j = 0; j < N; ++j) {
      EXPECT_EQ(packed_values[i][j], matrix((R + j) % matrix.rows(), i));
    }
  }
}

TEST_F(MatrixPackingTest, PackRowVerticallyWithExtensionField) {
  constexpr size_t N = PackedBabyBear::N;
  constexpr size_t R = 3;

  Matrix<BabyBear4> matrix = Matrix<BabyBear4>::Random(N, N);
  std::vector<PackedBabyBear> packed_values =
      PackRowVertically<PackedBabyBear>(matrix, R);
  ASSERT_EQ(packed_values.size(), 4 * N);
  for (size_t i = 0; i < packed_values.size(); ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t col = i / 4;
      size_t idx = i % 4;
      EXPECT_EQ(packed_values[i][j], matrix((R + j) % matrix.rows(), col)[idx]);
    }
  }
}

}  // namespace tachyon::math
