#include "third_party/eigen3/Eigen/LU"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::math {

class MatrixInverseTest : public FiniteFieldTest<GF7> {};

TEST_F(MatrixInverseTest, Inverse) {
  for (size_t i = 1; i < 10; ++i) {
    Matrix<GF7> matrix = Matrix<GF7>::Random(i, i);
    if (!matrix.determinant().IsZero()) {
      Matrix<GF7> inverse = matrix.inverse();
      EXPECT_TRUE(matrix * inverse == Matrix<GF7>::Identity(i, i));
    }
  }
}

}  // namespace tachyon::math
