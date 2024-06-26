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
  EXPECT_EQ(point, point);
  EXPECT_NE(point, point2);
}

TEST(Point2Test, ToString) {
  EXPECT_EQ(Point2GF7(GF7(1), GF7(2)).ToString(), "(1, 2)");
}

TEST(Point2Test, ToHexString) {
  EXPECT_EQ(Point2GF7(GF7(1), GF7(2)).ToHexString(), "(0x1, 0x2)");
}

TEST(Point2Test, Copyable) {
  Point2GF7 expected(GF7(1), GF7(2));

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  Point2GF7 value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST(Point2Test, JsonValueConverter) {
  Point2GF7 expected_point(GF7(1), GF7(2));
  std::string expected_json = R"({"x":1,"y":2})";

  Point2GF7 p;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &p, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(p, expected_point);

  std::string json = base::WriteToJson(p);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
