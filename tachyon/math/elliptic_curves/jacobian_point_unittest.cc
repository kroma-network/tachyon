#include "tachyon/math/elliptic_curves/jacobian_point.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/curve_config.h"

namespace tachyon::math {

namespace {

class JacobianPointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { test::JacobianPoint::Curve::Init(); }
};

}  // namespace

TEST_F(JacobianPointTest, Copyable) {
  test::JacobianPoint expected = test::JacobianPoint::Random();
  test::JacobianPoint value;

  base::VectorBuffer write_buf;
  write_buf.Write(expected);

  write_buf.set_buffer_offset(0);
  write_buf.Read(&value);

  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::math
