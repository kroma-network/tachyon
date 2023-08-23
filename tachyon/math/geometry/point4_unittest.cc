#include "tachyon/math/geometry/point4.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

using Point4GF7 = Point4<GF7>;

}  // namespace

TEST(Point4Test, Construct) {
  Point4GF7 point;
  EXPECT_EQ(point.x, GF7(0));
  EXPECT_EQ(point.y, GF7(0));
  EXPECT_EQ(point.z, GF7(0));
  EXPECT_EQ(point.w, GF7(0));

  point = Point4GF7(GF7(1), GF7(2), GF7(3), GF7(4));
  EXPECT_EQ(point.x, GF7(1));
  EXPECT_EQ(point.y, GF7(2));
  EXPECT_EQ(point.z, GF7(3));
  EXPECT_EQ(point.w, GF7(4));
}

TEST(Point4Test, EqualityOperators) {
  Point4GF7 point(GF7(1), GF7(2), GF7(3), GF7(4));
  Point4GF7 point2(GF7(4), GF7(5), GF7(6), GF7(0));
  EXPECT_TRUE(point == point);
  EXPECT_TRUE(point != point2);
}

TEST(Point4Test, ToString) {
  EXPECT_EQ(Point4GF7(GF7(1), GF7(2), GF7(3), GF7(4)).ToString(),
            "(1, 2, 3, 4)");
}

TEST(Point4Test, ToHexString) {
  EXPECT_EQ(Point4GF7(GF7(1), GF7(2), GF7(3), GF7(4)).ToHexString(),
            "(0x1, 0x2, 0x3, 0x4)");
}

}  // namespace tachyon::math
