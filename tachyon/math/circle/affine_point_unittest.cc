#include "gtest/gtest.h"

#include "tachyon/math/circle/m31/g1.h"
#include "tachyon/math/circle/m31/g4.h"

namespace tachyon::math {

template <typename AffinePointType>
class AffinePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { AffinePointType::Circle::Init(); }
};

using AffinePointTypes = testing::Types<m31::G1AffinePoint, m31::G4AffinePoint>;
TYPED_TEST_SUITE(AffinePointTest, AffinePointTypes);

TYPED_TEST(AffinePointTest, IsZero) {
  using AffinePoint = TypeParam;
  using BaseField = typename AffinePoint::BaseField;

  EXPECT_TRUE(AffinePoint::Zero().IsZero());
  EXPECT_FALSE(AffinePoint(BaseField::Zero(), BaseField::One()).IsZero());
}

TYPED_TEST(AffinePointTest, Generator) {
  using AffinePoint = TypeParam;
  using Circle = typename AffinePoint::Circle;

  EXPECT_EQ(
      AffinePoint::Generator(),
      AffinePoint(Circle::Config::kGenerator.x, Circle::Config::kGenerator.y));
}

TYPED_TEST(AffinePointTest, Order) {
  using AffinePoint = TypeParam;

  AffinePoint r = AffinePoint::Random();
  EXPECT_TRUE(r.ScalarMul(AffinePoint::ScalarField::Config::kModulus).IsZero());
}

TYPED_TEST(AffinePointTest, Random) {
  using AffinePoint = TypeParam;

  bool success = false;
  AffinePoint r = AffinePoint::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != AffinePoint::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
  EXPECT_TRUE(r.IsOnCircle());
}

TYPED_TEST(AffinePointTest, EqualityOperators) {
  using AffinePoint = TypeParam;
  using BaseField = typename AffinePoint::BaseField;

  AffinePoint p(BaseField::Random(), BaseField::Random());
  AffinePoint p2(BaseField::Random(), BaseField::Random());
  EXPECT_EQ(p, p);
  EXPECT_NE(p, p2);
}

TYPED_TEST(AffinePointTest, Conjugate) {
  using AffinePoint = TypeParam;

  AffinePoint p = AffinePoint::Random();
  EXPECT_EQ(p.Conjugate(), AffinePoint(p.x(), -p.y()));
}

TYPED_TEST(AffinePointTest, Antipode) {
  using AffinePoint = TypeParam;

  AffinePoint p = AffinePoint::Random();
  EXPECT_EQ(p.Antipode(), AffinePoint(-p.x(), -p.y()));
}

TYPED_TEST(AffinePointTest, AdditiveGroupOperators) {
  using AffinePoint = TypeParam;

  AffinePoint a = AffinePoint::Random();
  AffinePoint b = AffinePoint::Random();
  AffinePoint c = a + b;

  EXPECT_EQ(c - a, b);
  EXPECT_EQ(c - b, a);
  AffinePoint doubled = a.Double();
  EXPECT_EQ(doubled, a + a);
  AffinePoint a_tmp = a;
  EXPECT_EQ(a_tmp.DoubleInPlace(), doubled);

  EXPECT_EQ(a + AffinePoint::Zero(), a);
  a_tmp = a;
  a_tmp.NegateInPlace();
  EXPECT_EQ(a_tmp, -a);
}

}  // namespace tachyon::math
