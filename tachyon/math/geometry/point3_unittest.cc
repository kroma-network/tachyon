#include "tachyon/math/geometry/point3.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

using Point3GF7 = Point3<GF7>;

}  // namespace

TEST(Point3Test, Construct) {
  Point3GF7 point;
  EXPECT_EQ(point.x, GF7(0));
  EXPECT_EQ(point.y, GF7(0));
  EXPECT_EQ(point.z, GF7(0));

  point = Point3GF7(GF7(1), GF7(2), GF7(3));
  EXPECT_EQ(point.x, GF7(1));
  EXPECT_EQ(point.y, GF7(2));
  EXPECT_EQ(point.z, GF7(3));
}

TEST(Point3Test, EqualityOperators) {
  Point3GF7 point(GF7(1), GF7(2), GF7(3));
  Point3GF7 point2(GF7(4), GF7(5), GF7(6));
  EXPECT_TRUE(point == point);
  EXPECT_TRUE(point != point2);
}

TEST(Point3Test, ToString) {
  EXPECT_EQ(Point3GF7(GF7(1), GF7(2), GF7(3)).ToString(), "(1, 2, 3)");
}

TEST(Point3Test, ToHexString) {
  EXPECT_EQ(Point3GF7(GF7(1), GF7(2), GF7(3)).ToHexString(), "(0x1, 0x2, 0x3)");
}

TEST(Point3Test, Copyable) {
  Point3GF7 expected(GF7(1), GF7(2), GF7(3));

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  Point3GF7 value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST(Point3Test, JsonValueConverter) {
  Point3GF7 expected_point(GF7(1), GF7(2), GF7(3));
  std::string expected_json =
      R"({"x":{"value":"0x1"},"y":{"value":"0x2"},"z":{"value":"0x3"}})";

  Point3GF7 p;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &p, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(p, expected_point);

  std::string json = base::WriteToJson(p);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
