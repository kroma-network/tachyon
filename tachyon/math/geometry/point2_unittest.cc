#include "tachyon/math/geometry/point2.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

namespace {

using Point2i = Point2<int>;

}  // namespace

TEST(Point2Test, Construct) {
  Point2i point;
  EXPECT_EQ(point.x, 0);
  EXPECT_EQ(point.y, 0);

  point = Point2i(1, 2);
  EXPECT_EQ(point.x, 1);
  EXPECT_EQ(point.y, 2);
}

TEST(Point2Test, EqualityOperators) {
  Point2i point(1, 2);
  Point2i point2(4, 5);
  EXPECT_TRUE(point == point);
  EXPECT_TRUE(point != point2);
}

TEST(Point2Test, ToString) { EXPECT_EQ(Point2i(1, 2).ToString(), "(1, 2)"); }

}  // namespace math
}  // namespace tachyon
