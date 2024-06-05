#include "tachyon/math/matrix/matrix_utils.h"

#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::math {

class MatrixUtilsTest : public FiniteFieldTest<GF7> {};

TEST_F(MatrixUtilsTest, Circulant) {
  math::Matrix<GF7> circulant =
      math::MakeCirculant(math::Vector<GF7>{{GF7(2), GF7(3), GF7(4)}});
  math::Matrix<GF7> expected{{
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

  Matrix<BabyBear> matrix = Matrix<BabyBear>::Random(2 * N, 2 * N);
  std::vector<BabyBear> remaining_values;
  std::vector<PackedBabyBear> packed_values =
      PackRowHorizontally<PackedBabyBear>(matrix, R, remaining_values);
  ASSERT_TRUE(remaining_values.empty());
  ASSERT_EQ(packed_values.size(), 2);
  for (size_t i = 0; i < packed_values.size(); ++i) {
    for (size_t j = 0; j < N; ++j) {
      EXPECT_EQ(packed_values[i][j], matrix(R, i * N + j));
    }
  }

  matrix = Matrix<BabyBear>::Random(2 * N - 1, 2 * N - 1);
  remaining_values.clear();
  packed_values =
      PackRowHorizontally<PackedBabyBear>(matrix, R, remaining_values);
  ASSERT_EQ(remaining_values.size(), N - 1);
  ASSERT_EQ(packed_values.size(), 1);
  for (size_t i = 0; i < remaining_values.size(); ++i) {
    EXPECT_EQ(remaining_values[i], matrix(R, packed_values.size() * N + i));
  }
  for (size_t i = 0; i < packed_values.size(); ++i) {
    for (size_t j = 0; j < N; ++j) {
      EXPECT_EQ(packed_values[i][j], matrix(R, i * N + j));
    }
  }
}

TEST_F(MatrixPackingTest, PackRowVertically) {
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

}  // namespace tachyon::math
