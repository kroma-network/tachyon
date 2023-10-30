#include "tachyon/math/elliptic_curves/point_xyzz.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace tachyon::math {

namespace {

class PointXYZZTest : public testing::Test {
 public:
  static void SetUpTestSuite() { test::G1Curve::Init(); }
};

}  // namespace

TEST_F(PointXYZZTest, Copyable) {
  test::PointXYZZ expected = test::PointXYZZ::Random();
  test::PointXYZZ value;

  base::VectorBuffer write_buf;
  write_buf.Write(expected);

  write_buf.set_buffer_offset(0);
  write_buf.Read(&value);

  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::math
