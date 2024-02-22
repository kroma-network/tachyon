#include "tachyon/math/base/semigroups.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/msm/test/variable_base_msm_test_set.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

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
  static void SetUpTestSuite() {
    test::G1Curve::Init();

#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
    size_t thread_nums = 1;
#endif
    test_set_ = VariableBaseMSMTestSet<test::AffinePoint>::Random(
        thread_nums * 5, VariableBaseMSMMethod::kNone);
  }

 protected:
  static VariableBaseMSMTestSet<test::AffinePoint> test_set_;
};

VariableBaseMSMTestSet<test::AffinePoint> MultiScalarMulTest::test_set_;

}  // namespace

// scalar: s
// bases: [G₀, G₁, ..., Gₙ₋₁]
// return: [sG₀, sG₁, ..., sGₙ₋₁]
TEST_F(MultiScalarMulTest, SingleScalarMultiBases) {
  const GF7& scalar = test_set_.scalars[0];
  std::vector<test::JacobianPoint> expected = base::Map(
      test_set_.bases,
      [&scalar](const test::AffinePoint& base) { return base * scalar; });
  {
    std::vector<test::JacobianPoint> actual;
    actual.resize(test_set_.bases.size());
    ASSERT_TRUE(test::AffinePoint::MultiScalarMul(test_set_.scalars[0],
                                                  test_set_.bases, &actual));
    EXPECT_EQ(actual, expected);
  }
  {
    std::vector<test::JacobianPoint> actual;
    actual.resize(test_set_.bases.size());
    ASSERT_TRUE(test::AffinePoint::MultiScalarMul(test_set_.scalars[0],
                                                  test_set_.bases, &actual));
    EXPECT_EQ(actual, expected);
  }
}

// scalars: [s₀, s₁, ..., sₙ₋₁]
// base: G
// return: [s₀G, s₁G, ..., sₙ₋₁G]
TEST_F(MultiScalarMulTest, MultiScalarsSingleBase) {
  const test::AffinePoint& base = test_set_.bases[0];
  std::vector<test::JacobianPoint> expected = base::Map(
      test_set_.scalars, [&base](const GF7& scalar) { return base * scalar; });
  {
    std::vector<test::JacobianPoint> actual;
    actual.resize(test_set_.scalars.size());
    ASSERT_TRUE(test::AffinePoint::MultiScalarMul(test_set_.scalars,
                                                  test_set_.bases[0], &actual));
    EXPECT_EQ(actual, expected);
  }
  {
    std::vector<test::JacobianPoint> actual;
    actual.resize(test_set_.bases.size());
    ASSERT_TRUE(test::AffinePoint::MultiScalarMul(test_set_.scalars,
                                                  test_set_.bases[0], &actual));
    EXPECT_EQ(actual, expected);
  }
}

// scalars: [s₀, s₁, ..., sₙ₋₁]
// bases: [G₀, G₁, ..., Gₙ₋₁]
// return: [s₀G₀, s₁G₁, ..., sₙ₋₁Gₙ₋₁]
TEST_F(MultiScalarMulTest, MultiScalarsMultiBases) {
  std::vector<test::JacobianPoint> expected = base::Map(
      test_set_.scalars,
      [](size_t i, const GF7& scalar) { return test_set_.bases[i] * scalar; });
  {
    std::vector<test::JacobianPoint> actual;
    actual.resize(test_set_.scalars.size());
    ASSERT_TRUE(test::AffinePoint::MultiScalarMul(test_set_.scalars,
                                                  test_set_.bases, &actual));
    EXPECT_EQ(actual, expected);
  }
  {
    std::vector<test::JacobianPoint> actual;
    actual.resize(test_set_.bases.size());
    ASSERT_TRUE(test::AffinePoint::MultiScalarMul(test_set_.scalars,
                                                  test_set_.bases, &actual));
    EXPECT_EQ(actual, expected);
  }
}

}  // namespace tachyon::math
