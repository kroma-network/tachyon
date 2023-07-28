#include "tachyon/math/base/groups.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tachyon::math {

TEST(GroupsTest, Div) {
  class Int : public MultiplicativeGroup<Int> {
   public:
    Int() : value_(0) {}
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Mul, (const Int& other), (const));
    MOCK_METHOD(Int, Inverse, (), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_;
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
    Int() : value_(0) {}
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Div, (const Int& other), (const));
    MOCK_METHOD(Int, Mul, (const Int& other), (const));
    MOCK_METHOD(Int, Inverse, (), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Div(b)).Times(testing::Exactly(1));
  EXPECT_CALL(a, Mul(testing::_)).Times(testing::Exactly(0));
  EXPECT_CALL(b, Inverse()).Times(testing::Exactly(0));

  Int c = a / b;
  static_cast<void>(c);
}

TEST(GroupsTest, Sub) {
  class Int : public AdditiveGroup<Int> {
   public:
    Int() : value_(0) {}
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Add, (const Int& other), (const));
    MOCK_METHOD(Int, Negative, (), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_;
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
    Int() : value_(0) {}
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Sub, (const Int& other), (const));
    MOCK_METHOD(Int, Add, (const Int& other), (const));
    MOCK_METHOD(Int, Inverse, (), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_;
  };

  Int a(3);
  Int b(4);
  EXPECT_CALL(a, Sub(b)).Times(testing::Exactly(1));
  EXPECT_CALL(a, Add(testing::_)).Times(testing::Exactly(0));
  EXPECT_CALL(b, Inverse()).Times(testing::Exactly(0));

  Int c = a - b;
  static_cast<void>(c);
}

}  // namespace tachyon::math
