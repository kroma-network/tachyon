#include "tachyon/math/geometry/dimensions.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"

namespace tachyon::math {

TEST(DimensionsTest, Construct) {
  Dimensions dimensions;
  EXPECT_EQ(dimensions.width, 0);
  EXPECT_EQ(dimensions.height, 0);

  dimensions = Dimensions(1, 2);
  EXPECT_EQ(dimensions.width, 1);
  EXPECT_EQ(dimensions.height, 2);
}

TEST(DimensionsTest, EqualityOperators) {
  Dimensions dimensions(1, 2);
  Dimensions dimensions2(4, 5);
  EXPECT_EQ(dimensions, dimensions);
  EXPECT_NE(dimensions, dimensions2);
}

TEST(DimensionsTest, ToString) {
  EXPECT_EQ(Dimensions(1, 2).ToString(), "(1, 2)");
}

TEST(DimensionsTest, Copyable) {
  Dimensions expected(1, 2);

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  Dimensions value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST(DimensionsTest, JsonValueConverter) {
  Dimensions expected_dimensions(1, 2);
  std::string expected_json = R"({"width":1,"height":2})";

  Dimensions dimensions;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &dimensions, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(dimensions, expected_dimensions);

  std::string json = base::WriteToJson(dimensions);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
