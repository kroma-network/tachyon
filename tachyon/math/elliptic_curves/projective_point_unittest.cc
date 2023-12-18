#include "tachyon/math/elliptic_curves/projective_point.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace tachyon::math {

namespace {

class ProjectivePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { test::G1Curve::Init(); }
};

}  // namespace

TEST_F(ProjectivePointTest, Copyable) {
  test::ProjectivePoint expected = test::ProjectivePoint::Random();
  test::ProjectivePoint value;

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Write(expected));

  write_buf.set_buffer_offset(0);
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::math
