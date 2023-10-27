#include "tachyon/math/base/groups.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

TEST(GroupsTest, Div) {
  class Int : public MultiplicativeGroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Mul, (const Int& other), (const));
    MOCK_METHOD(Int, Inverse, (), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Mul(testing::_)).Times(testing::Exactly(1));
  EXPECT_CALL(b, Inverse()).Times(testing::Exactly(1));

  Int c = a / b;
  static_cast<void>(c);
}

TEST(GroupsTest, DivOverMul) {
  class Int : public MultiplicativeGroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Div, (const Int& other), (const));
    MOCK_METHOD(Int, Mul, (const Int& other), (const));
    MOCK_METHOD(Int, Inverse, (), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Div(b)).Times(testing::Exactly(1));
  EXPECT_CALL(a, Mul(testing::_)).Times(testing::Exactly(0));
  EXPECT_CALL(b, Inverse()).Times(testing::Exactly(0));

  Int c = a / b;
  static_cast<void>(c);
}

TEST(GroupsTest, InverseOverride) {
  class IntInverse {
   public:
    IntInverse() = default;
    explicit IntInverse(int denominator) : denominator_(denominator) {}
    IntInverse(const IntInverse& other) : denominator_(other.denominator_) {}

    bool operator==(const IntInverse& other) const {
      return denominator_ == other.denominator_;
    }

   private:
    int denominator_ = 0;
  };

  class Int : public MultiplicativeGroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    IntInverse Inverse() const { return IntInverse(value_); }

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  EXPECT_EQ(a.Inverse(), IntInverse(3));
}

TEST(GroupsTest, BatchInverse) {
  math::GF7::Init();
#if defined(TACHYON_HAS_OPENMP)
  size_t size = size_t{1} << (static_cast<size_t>(omp_get_max_threads()) /
                              GF7::kParallelBatchInverseDivisorThreshold);
#else
  size_t size = 5;
#endif
  // GF7 is MultiplicativeGroup because it satisfies the conditions of Field.
  std::vector<GF7> groups =
      base::CreateVector(size, []() { return GF7::Random(); });
  std::vector<GF7> inverses;
  inverses.resize(groups.size());
  ASSERT_TRUE(GF7::BatchInverse(groups, &inverses));
  for (size_t i = 0; i < groups.size(); ++i) {
    if (groups[i].IsZero()) {
      EXPECT_TRUE(inverses[i].IsZero());
    } else {
      EXPECT_TRUE((inverses[i] * groups[i]).IsOne());
    }
  }

  ASSERT_TRUE(GF7::BatchInverseInPlace(groups));
  EXPECT_EQ(groups, inverses);
}

TEST(GroupsTest, Sub) {
  class Int : public AdditiveGroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Add, (const Int& other), (const));
    MOCK_METHOD(Int, Negative, (), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Add(testing::_)).Times(testing::Exactly(1));
  EXPECT_CALL(b, Negative()).Times(testing::Exactly(1));

  Int c = a - b;
  static_cast<void>(c);
}

TEST(GroupsTest, SubOverAdd) {
  class Int : public AdditiveGroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Sub, (const Int& other), (const));
    MOCK_METHOD(Int, Add, (const Int& other), (const));
    MOCK_METHOD(Int, Inverse, (), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Sub(b)).Times(testing::Exactly(1));
  EXPECT_CALL(a, Add(testing::_)).Times(testing::Exactly(0));
  EXPECT_CALL(b, Inverse()).Times(testing::Exactly(0));

  Int c = a - b;
  static_cast<void>(c);
}

TEST(GroupsTest, NegativeOverride) {
  class IntNegative {
   public:
    IntNegative() = default;
    explicit IntNegative(int value) : value_(value) {}
    IntNegative(const IntNegative& other) : value_(other.value_) {}

    bool operator==(const IntNegative& other) const {
      return value_ == other.value_;
    }

   private:
    int value_ = 0;
  };

  class Int : public AdditiveGroup<Int> {
   public:
    Int() = default;
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    IntNegative Negative() const { return IntNegative(-value_); }

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_ = 0;
  };

  Int a(3);
  EXPECT_EQ(-a, IntNegative(-3));
}

}  // namespace tachyon::math
