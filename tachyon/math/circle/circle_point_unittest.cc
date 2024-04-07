#include "gtest/gtest.h"

#include "tachyon/math/circle/stark/g1.h"

namespace tachyon::math {

template <typename CirclePointType>
class CirclePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { CirclePointType::Circle::Init(); }
};

using CirclePointTypes = testing::Types<stark::G1CirclePoint>;
TYPED_TEST_SUITE(CirclePointTest, CirclePointTypes);

TYPED_TEST(CirclePointTest, IsZero) {
  using CirclePoint = TypeParam;
  using BaseField = typename CirclePoint::BaseField;

  EXPECT_TRUE(CirclePoint::Zero().IsZero());
  EXPECT_FALSE(CirclePoint(BaseField::Zero(), BaseField::One()).IsZero());
}

TYPED_TEST(CirclePointTest, Generator) {
  using CirclePoint = TypeParam;
  using Circle = typename CirclePoint::Circle;

  EXPECT_EQ(
      CirclePoint::Generator(),
      CirclePoint(Circle::Config::kGenerator.x, Circle::Config::kGenerator.y));
}

TYPED_TEST(CirclePointTest, Order) {
  using CirclePoint = TypeParam;

  CirclePoint r = CirclePoint::Random();
  EXPECT_TRUE(r.ScalarMul(CirclePoint::ScalarField::Config::kModulus).IsZero());
}

TYPED_TEST(CirclePointTest, Random) {
  using CirclePoint = TypeParam;

  bool success = false;
  CirclePoint r = CirclePoint::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != CirclePoint::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
  EXPECT_TRUE(r.IsOnCircle());
}

TYPED_TEST(CirclePointTest, EqualityOperators) {
  using CirclePoint = TypeParam;
  using BaseField = typename CirclePoint::BaseField;

  CirclePoint p(BaseField::Random(), BaseField::Random());
  CirclePoint p2(BaseField::Random(), BaseField::Random());
  EXPECT_TRUE(p == p);
  EXPECT_TRUE(p != p2);
}

TYPED_TEST(CirclePointTest, Conjugate) {
  using CirclePoint = TypeParam;

  CirclePoint p = CirclePoint::Random();
  EXPECT_EQ(p.Conjugate(), CirclePoint(p.x(), -p.y()));
}

TYPED_TEST(CirclePointTest, Antipode) {
  using CirclePoint = TypeParam;

  CirclePoint p = CirclePoint::Random();
  EXPECT_EQ(p.Antipode(), CirclePoint(-p.x(), -p.y()));
}

TYPED_TEST(CirclePointTest, AdditiveGroupOperators) {
  using CirclePoint = TypeParam;

  CirclePoint a = CirclePoint::Random();
  CirclePoint b = CirclePoint::Random();
  CirclePoint c = a + b;

  EXPECT_EQ(c - a, b);
  EXPECT_EQ(c - b, a);
  EXPECT_EQ(a.Double(), a + a);

  EXPECT_EQ(a + CirclePoint::Zero(), a);
  CirclePoint a_tmp = a;
  a_tmp.NegateInPlace();
  EXPECT_EQ(a_tmp, -a);
}

}  // namespace tachyon::math
