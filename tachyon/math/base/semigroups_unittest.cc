#include "tachyon/math/base/semigroups.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(SemigroupsTest, Mul) {
  class Int : public MultiplicativeSemigroup<Int> {
   public:
    Int() : value_(0) {}
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Mul, (const Int& other), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_;
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
    Int() : value_(0) {}
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Mul, (const Int& other), (const));
    MOCK_METHOD(Int&, MulInPlace, (const Int& other));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_;
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
    Int() : value_(0) {}
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Add, (const Int& other), (const));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_;
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
    Int() : value_(0) {}
    explicit Int(int value) : value_(value) {}
    Int(const Int& other) : value_(other.value_) {}

    MOCK_METHOD(Int, Add, (const Int& other), (const));
    MOCK_METHOD(Int&, AddInPlace, (const Int& other));

    bool operator==(const Int& other) const { return value_ == other.value_; }

   private:
    int value_;
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

}  // namespace math
}  // namespace tachyon
