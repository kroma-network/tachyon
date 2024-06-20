#include "tachyon/math/matrix/matrix_types.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

class MatrixTypesTest : public FiniteFieldTest<GF7> {};

TEST_F(MatrixTypesTest, CopyableDynamicMatrix) {
  Matrix<GF7> expected{
      {GF7(0), GF7(1), GF7(2)},
      {GF7(3), GF7(4), GF7(5)},
      {GF7(6), GF7(0), GF7(1)},
  };

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  {
    write_buf.set_buffer_offset(0);
    Matrix<GF7, 2, 3> value;
    ASSERT_FALSE(write_buf.Read(&value));
  }
  {
    write_buf.set_buffer_offset(0);
    Matrix<GF7, 3, 2> value;
    ASSERT_FALSE(write_buf.Read(&value));
  }
  {
    write_buf.set_buffer_offset(0);
    Matrix<GF7, 3, 3> value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(value, expected);
  }
  {
    write_buf.set_buffer_offset(0);
    Matrix<GF7> value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(value, expected);
  }
}

TEST_F(MatrixTypesTest, Copyable3x3Matrix) {
  Matrix<GF7, 3, 3> expected{
      {GF7(0), GF7(1), GF7(2)},
      {GF7(3), GF7(4), GF7(5)},
      {GF7(6), GF7(0), GF7(1)},
  };

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  {
    write_buf.set_buffer_offset(0);
    Matrix<GF7, 2, 3> value;
    ASSERT_FALSE(write_buf.Read(&value));
  }
  {
    write_buf.set_buffer_offset(0);
    Matrix<GF7, 3, 2> value;
    ASSERT_FALSE(write_buf.Read(&value));
  }
  {
    write_buf.set_buffer_offset(0);
    Matrix<GF7, 3, 3> value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(value, expected);
  }
  {
    write_buf.set_buffer_offset(0);
    Matrix<GF7> value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(value, expected);
  }
}

TEST_F(MatrixTypesTest, CopyableDynamicDiagonalMatrix) {
  DiagonalMatrix<GF7> expected{{GF7(1), GF7(2), GF7(3)}};

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  {
    write_buf.set_buffer_offset(0);
    DiagonalMatrix<GF7, 2> value;
    ASSERT_FALSE(write_buf.Read(&value));
  }
  {
    write_buf.set_buffer_offset(0);
    DiagonalMatrix<GF7, 3> value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(value.diagonal(), expected.diagonal());
  }
  {
    write_buf.set_buffer_offset(0);
    DiagonalMatrix<GF7> value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(value.diagonal(), expected.diagonal());
  }
}

TEST_F(MatrixTypesTest, Copyable3x3DiagonalMatrix) {
  DiagonalMatrix<GF7, 3> expected{GF7(1), GF7(2), GF7(3)};

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  {
    write_buf.set_buffer_offset(0);
    DiagonalMatrix<GF7, 2> value;
    ASSERT_FALSE(write_buf.Read(&value));
  }
  {
    write_buf.set_buffer_offset(0);
    DiagonalMatrix<GF7, 3> value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(value.diagonal(), expected.diagonal());
  }
  {
    write_buf.set_buffer_offset(0);
    DiagonalMatrix<GF7> value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(value.diagonal(), expected.diagonal());
  }
}

}  // namespace tachyon::math
