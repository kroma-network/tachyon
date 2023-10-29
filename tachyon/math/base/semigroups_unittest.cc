#include "tachyon/math/base/semigroups.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/test/curve_config.h"

namespace tachyon::math {

TEST(SemigroupsTest, Mul) {
  class Int : public MultiplicativeSemigroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Mul, (const Int& other), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Mul(b)).Times(testing::Exactly(1));

  Int c = a * b;
  static_cast<void>(c);
}

TEST(SemigroupsTest, MulOverMulInPlace) {
  class Int : public MultiplicativeSemigroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Mul, (const Int& other), (const));
    MOCK_METHOD(Int&, MulInPlace, (const Int& other));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Mul(b)).Times(testing::Exactly(1));

  Int c = a * b;
  static_cast<void>(c);

  EXPECT_CALL(c, Mul(c)).Times(testing::Exactly(1));

  Int d = c.Square();
  static_cast<void>(d);
}

TEST(SemigroupsTest, Add) {
  class Int : public AdditiveSemigroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Add, (const Int& other), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Add(b)).Times(testing::Exactly(1));

  Int c = a + b;
  static_cast<void>(c);
}

TEST(SemigroupsTest, AddOverAddInPlace) {
  class Int : public AdditiveSemigroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Add, (const Int& other), (const));
    MOCK_METHOD(Int&, AddInPlace, (const Int& other));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Add(b)).Times(testing::Exactly(1));

  Int c = a + b;
  static_cast<void>(c);

  EXPECT_CALL(c, Add(c)).Times(testing::Exactly(1));

  Int d = c.Double();
  static_cast<void>(d);
}

namespace {

class MultiScalarMulTest : public testing::Test {
 public:
  using BaseTy = test::AffinePoint;
  using BaseField = typename BaseTy::BaseField;
  using ReturnTy = typename internal::AdditiveSemigroupTraits<BaseTy>::ReturnTy;

  static void SetUpTestSuite() { test::AffinePoint::Curve::Init(); }
};

}  // namespace

// scalar: s
// bases: [G₀, G₁, ..., Gₙ₋₁]
// return: [sG₀, sG₁, ..., sGₙ₋₁]
TEST_F(MultiScalarMulTest, SingleScalarMultiBases) {
  BaseField s = BaseField(3);
  std::vector<BaseTy> bases = {BaseTy::Random(), BaseTy::Random(),
                               BaseTy::Random()};
  std::vector<ReturnTy> expected;
  for (const BaseTy& base : bases) {
    expected.push_back(base.ScalarMul(s.ToBigInt()));
  }
  std::vector<ReturnTy> actual = BaseTy::MultiScalarMul(s, bases);

  EXPECT_EQ(actual, expected);
}

// scalars: [s₀, s₁, ..., sₙ₋₁]
// base: G
// return: [s₀G, s₁G, ..., sₙ₋₁G]
TEST_F(MultiScalarMulTest, MultiScalarsSingleBase) {
  std::vector<BaseField> scalars = {BaseField(3), BaseField(4), BaseField(5)};
  BaseTy base = BaseTy::Random();
  std::vector<ReturnTy> expected;
  for (const BaseField& scalar : scalars) {
    expected.push_back(base.ScalarMul(scalar.ToBigInt()));
  }
  std::vector<ReturnTy> actual = BaseTy::MultiScalarMul(scalars, base);
  EXPECT_EQ(actual, expected);
}

// scalars: [s₀, s₁, ..., sₙ₋₁]
// bases: [G₀, G₁, ..., Gₙ₋₁]
// return: [s₀G₀, s₁G₁, ..., sₙ₋₁Gₙ₋₁]
TEST_F(MultiScalarMulTest, MultiScalarsMultiBases) {
  std::vector<BaseField> scalars = {BaseField(3), BaseField(4), BaseField(5)};
  std::vector<BaseTy> bases = {BaseTy::Random(), BaseTy::Random(),
                               BaseTy::Random()};
  std::vector<ReturnTy> expected;
  for (size_t i = 0; i < scalars.size(); ++i) {
    expected.push_back(bases[i].ScalarMul(scalars[i].ToBigInt()));
  }
  std::vector<ReturnTy> actual = BaseTy::MultiScalarMul(scalars, bases);
  EXPECT_EQ(actual, expected);
}

}  // namespace tachyon::math
