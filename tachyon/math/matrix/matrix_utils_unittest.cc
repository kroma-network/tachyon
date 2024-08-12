#include "tachyon/math/matrix/matrix_utils.h"

#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
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

TEST_F(MatrixPackingTest, SplitMat) {
  Matrix<BabyBear> matrix = Matrix<BabyBear>::Random(10, 10);
  std::vector<Eigen::Block<Matrix<BabyBear>>> result = SplitMat(4, matrix);
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], matrix.block(i, 0, 7, matrix.cols()));
  }
}

}  // namespace tachyon::math
