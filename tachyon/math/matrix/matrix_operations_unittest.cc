#include "tachyon/math/matrix/matrix_operations.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

class MatrixOperationsTest : public FiniteFieldTest<GF7> {};

TEST_F(MatrixOperationsTest, MulMatVecSerialWithVector) {
  Matrix<GF7> matrix = Matrix<GF7>::Random(2, 2);
  Vector<GF7> vector = Vector<GF7>::Random(2);

  EXPECT_EQ(matrix * vector, MulMatVecSerial(matrix, vector));
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
  Matrix<GF7> matrix = Matrix<GF7>::Random(3, 2);
  Matrix<GF7> matrix2 = Matrix<GF7>::Random(2, 5);

  EXPECT_EQ(matrix * matrix2, MulMatMatSerial(matrix, matrix2));
}

#if defined(TACHYON_HAS_OPENMP)
TEST_F(MatrixOperationsTest, MulMatVecWithVector) {
  Matrix<GF7> matrix = Matrix<GF7>::Random(100, 100);
  Vector<GF7> vector = Vector<GF7>::Random(100);

  EXPECT_EQ(matrix * vector, MulMatVec(matrix, vector));
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
  Matrix<GF7> matrix = Matrix<GF7>::Random(3, 2);
  Matrix<GF7> matrix2 = Matrix<GF7>::Random(2, 5);

  EXPECT_EQ(matrix * matrix2, MulMatMat(matrix, matrix2));
}
#endif

}  // namespace tachyon::math
