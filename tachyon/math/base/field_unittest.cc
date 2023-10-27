#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

class FieldTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::GF7::Init(); }
};

}  // namespace

TEST(FieldTest, SumOfProducts) {
  std::vector<GF7> a = {GF7(2), GF7(3), GF7(4)};
  std::vector<GF7> b = {GF7(1), GF7(2), GF7(3)};

  GF7 result = Field<GF7>::SumOfProducts(a, b);
  EXPECT_EQ(result, GF7(6));
}

}  // namespace tachyon::math
