#include "tachyon/math/matrix/matrix_operations.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/matrix/matrix_utils.h"

namespace tachyon::math {

class MatrixOperationsTest : public FiniteFieldTest<GF7> {};

TEST_F(MatrixOperationsTest, MulMatVecSerial) {
  constexpr size_t kRows = 3;
  constexpr size_t kCols = 2;

  Matrix<GF7, kRows, kCols> matrix = Matrix<GF7, kRows, kCols>::Random();
  Vector<GF7, kCols> vector = Vector<GF7, kCols>::Random();
  Vector<GF7, kRows> answer = matrix * vector;

  EXPECT_EQ(ToArray(answer),
            MulMatVecSerial(To2DArray(matrix), ToArray(vector)));
}

TEST_F(MatrixOperationsTest, MulMatVecSerialWithRowVector) {
  Matrix<GF7> matrix = Matrix<GF7>::Random(2, 2);
  RowVector<GF7> vector = RowVector<GF7>::Random(2);

  EXPECT_EQ(matrix * vector.transpose(), MulMatVecSerial(matrix, vector));
}

TEST_F(MatrixOperationsTest, MulMatVecSerialWithCoefficients) {
  Matrix<GF7> matrix = Matrix<GF7>::Random(2, 2);
  Matrix<GF7> matrix2 = Matrix<GF7>::Random(2, 2);

  EXPECT_EQ(matrix * matrix2.col(0), MulMatVecSerial(matrix, matrix2.col(0)));
  EXPECT_EQ(matrix * matrix2.row(0).transpose(),
            MulMatVecSerial(matrix, matrix2.row(0)));
}

TEST_F(MatrixOperationsTest, MulMatMatSerial) {
  constexpr size_t kRows = 3;
  constexpr size_t kCols = 2;
  constexpr size_t kCols2 = 5;

  Matrix<GF7, kRows, kCols> matrix = Matrix<GF7, kRows, kCols>::Random();
  Matrix<GF7, kCols, kCols2> matrix2 = Matrix<GF7, kCols, kCols2>::Random();
  Matrix<GF7, kRows, kCols2> answer = matrix * matrix2;

  EXPECT_EQ(To2DArray(answer),
            MulMatMat(To2DArray(matrix), To2DArray(matrix2)));
}

#if defined(TACHYON_HAS_OPENMP)
TEST_F(MatrixOperationsTest, MulMatVecWithVector) {
  Matrix<GF7> matrix = Matrix<GF7>::Random(100, 100);
  Vector<GF7> vector = Vector<GF7>::Random(100);

  EXPECT_EQ(matrix * vector, MulMatVec(matrix, vector));
}

TEST_F(MatrixOperationsTest, MulMatVec) {
  constexpr size_t kRows = 60;
  constexpr size_t kCols = 40;

  Matrix<GF7, kRows, kCols> matrix = Matrix<GF7, kRows, kCols>::Random();
  Vector<GF7, kCols> vector = Vector<GF7, kCols>::Random();
  Vector<GF7, kRows> answer = matrix * vector;

  EXPECT_EQ(ToArray(answer), MulMatVec(To2DArray(matrix), ToArray(vector)));
}

TEST_F(MatrixOperationsTest, MulMatVecWithRowVector) {
  Matrix<GF7> matrix = Matrix<GF7>::Random(100, 100);
  RowVector<GF7> vector = RowVector<GF7>::Random(100);

  EXPECT_EQ(matrix * vector.transpose(), MulMatVec(matrix, vector));
}

TEST_F(MatrixOperationsTest, MulMatVecWithCoefficients) {
  Matrix<GF7> matrix = Matrix<GF7>::Random(100, 100);
  Matrix<GF7> matrix2 = Matrix<GF7>::Random(100, 100);

  EXPECT_EQ(matrix * matrix2.col(0), MulMatVec(matrix, matrix2.col(0)));
  EXPECT_EQ(matrix * matrix2.row(0).transpose(),
            MulMatVec(matrix, matrix2.row(0)));
}

TEST_F(MatrixOperationsTest, MulMatMat) {
  constexpr size_t kRows = 60;
  constexpr size_t kCols = 40;
  constexpr size_t kCols2 = 50;

  Matrix<GF7, kRows, kCols> matrix = Matrix<GF7, kRows, kCols>::Random();
  Matrix<GF7, kCols, kCols2> matrix2 = Matrix<GF7, kCols, kCols2>::Random();
  Matrix<GF7, kRows, kCols2> answer = matrix * matrix2;

  EXPECT_EQ(To2DArray(answer),
            MulMatMat(To2DArray(matrix), To2DArray(matrix2)));
}
#endif

}  // namespace tachyon::math
