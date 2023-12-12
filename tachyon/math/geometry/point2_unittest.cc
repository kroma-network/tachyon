#include "tachyon/math/geometry/point2.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

using Point2GF7 = Point2<GF7>;

}  // namespace

TEST(Point2Test, Construct) {
  Point2GF7 point;
  EXPECT_EQ(point.x, GF7(0));
  EXPECT_EQ(point.y, GF7(0));

  point = Point2GF7(GF7(1), GF7(2));
  EXPECT_EQ(point.x, GF7(1));
  EXPECT_EQ(point.y, GF7(2));
}

TEST(Point2Test, EqualityOperators) {
  Point2GF7 point(GF7(1), GF7(2));
  Point2GF7 point2(GF7(4), GF7(5));
  EXPECT_TRUE(point == point);
  EXPECT_TRUE(point != point2);
}

TEST(Point2Test, ToString) {
  EXPECT_EQ(Point2GF7(GF7(1), GF7(2)).ToString(), "(1, 2)");
}

TEST(Point2Test, ToHexString) {
  EXPECT_EQ(Point2GF7(GF7(1), GF7(2)).ToHexString(), "(0x1, 0x2)");
}

TEST(Point2Test, Copyable) {
  Point2GF7 expected(GF7(1), GF7(2));
  Point2GF7 value;

  base::VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Write(expected));

  write_buf.set_buffer_offset(0);
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::math
