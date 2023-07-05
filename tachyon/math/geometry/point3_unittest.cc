#include "tachyon/math/geometry/point3.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

namespace {

using Point3i = Point3<int>;

}  // namespace

TEST(Point3Test, Construct) {
  Point3i point;
  EXPECT_EQ(point.x, 0);
  EXPECT_EQ(point.y, 0);
  EXPECT_EQ(point.z, 0);

  point = Point3i(1, 2, 3);
  EXPECT_EQ(point.x, 1);
  EXPECT_EQ(point.y, 2);
  EXPECT_EQ(point.z, 3);
}

TEST(Point3Test, EqualityOperators) {
  Point3i point(1, 2, 3);
  Point3i point2(4, 5, 6);
  EXPECT_TRUE(point == point);
  EXPECT_TRUE(point != point2);
}

TEST(Point3Test, ToString) {
  EXPECT_EQ(Point3i(1, 2, 3).ToString(), "(1, 2, 3)");
}

}  // namespace math
}  // namespace tachyon
