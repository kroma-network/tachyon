#include "tachyon/zk/base/value.h"

#include <tuple>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk {

TEST(ValueTest, Zero) {
  EXPECT_TRUE(Value<math::GF7>::Zero().IsZero());
  EXPECT_FALSE(Value<math::GF7>::One().IsZero());
}

TEST(ValueTest, One) {
  EXPECT_TRUE(Value<math::GF7>::One().IsOne());
  EXPECT_FALSE(Value<math::GF7>::Zero().IsOne());
}

TEST(ValueTest, IsNone) {
  EXPECT_TRUE(Value<math::GF7>::Unknown().IsNone());
  EXPECT_FALSE(Value<math::GF7>::Known(math::GF7(3)).IsNone());
}

TEST(ValueTest, AdditiveOperators) {
  math::GF7 a = math::GF7::Random();
  math::GF7 b = math::GF7::Random();

  struct {
    Value<math::GF7> a;
    Value<math::GF7> b;
    Value<math::GF7> sum;
    Value<math::GF7> amb;
    Value<math::GF7> bma;
  } tests[] = {
      {Value<math::GF7>::Known(a), Value<math::GF7>::Known(b),
       Value<math::GF7>::Known(a + b), Value<math::GF7>::Known(a - b),
       Value<math::GF7>::Known(b - a)},
      {Value<math::GF7>::Known(a), Value<math::GF7>::Unknown(),
       Value<math::GF7>::Unknown(), Value<math::GF7>::Unknown(),
       Value<math::GF7>::Unknown()},
  };

  for (const auto& test : tests) {
    if (test.a.IsNone() || test.b.IsNone()) {
      EXPECT_TRUE((test.a + test.b).IsNone());
      EXPECT_TRUE((test.b + test.a).IsNone());
      EXPECT_TRUE((test.a - test.b).IsNone());
      EXPECT_TRUE((test.b - test.a).IsNone());
    } else {
      EXPECT_EQ(test.a + test.b, test.sum);
      EXPECT_EQ(test.b + test.a, test.sum);
      EXPECT_EQ(test.a - test.b, test.amb);
      EXPECT_EQ(test.b - test.a, test.bma);
    }
  }
}

TEST(ValueTest, AdditiveGroupOperators) {
  math::GF7 a = math::GF7::Random();

  struct {
    Value<math::GF7> a;
    Value<math::GF7> neg;
    Value<math::GF7> dbl;
  } tests[] = {
      {
          Value<math::GF7>::Known(a),
          Value<math::GF7>::Known(-a),
          Value<math::GF7>::Known(a.Double()),
      },
      {
          Value<math::GF7>::Unknown(),
          Value<math::GF7>::Unknown(),
          Value<math::GF7>::Unknown(),
      },
  };

  for (auto& test : tests) {
    if (test.a.IsNone()) {
      EXPECT_TRUE(-test.a.IsNone());
      EXPECT_TRUE(test.a.NegateInPlace().IsNone());
      EXPECT_TRUE(test.a.Double().IsNone());
      EXPECT_TRUE(test.a.DoubleInPlace().IsNone());
    } else {
      EXPECT_EQ(-test.a, test.neg);
      Value<math::GF7> a_tmp = test.a;
      a_tmp.NegateInPlace();
      EXPECT_EQ(a_tmp, test.neg);

      EXPECT_EQ(test.a.Double(), test.dbl);
      a_tmp = test.a;
      a_tmp.DoubleInPlace();
      EXPECT_EQ(a_tmp, test.dbl);
    }
  }
}

TEST(ValueTest, MultiplicativeOperators) {
  math::GF7 a = math::GF7::Random();
  math::GF7 b = math::GF7::Random();
  while (a.IsZero()) {
    a = math::GF7::Random();
  }
  while (b.IsZero()) {
    b = math::GF7::Random();
  }

  struct {
    Value<math::GF7> a;
    Value<math::GF7> b;
    Value<math::GF7> mul;
    Value<math::GF7> adb;
    Value<math::GF7> bda;
  } tests[] = {
      {Value<math::GF7>::Known(a), Value<math::GF7>::Known(b),
       Value<math::GF7>::Known(a * b), Value<math::GF7>::Known(*(a / b)),
       Value<math::GF7>::Known(*(b / a))},
      {Value<math::GF7>::Known(a), Value<math::GF7>::Unknown(),
       Value<math::GF7>::Unknown(), Value<math::GF7>::Unknown(),
       Value<math::GF7>::Unknown()},
  };

  for (const auto& test : tests) {
    if (test.a.IsNone() || test.b.IsNone()) {
      EXPECT_TRUE((test.a * test.b).IsNone());
      EXPECT_TRUE((test.b * test.a).IsNone());
      EXPECT_TRUE((*(test.a / test.b)).IsNone());
      EXPECT_TRUE((*(test.b / test.a)).IsNone());
    } else {
      EXPECT_EQ(test.a * test.b, test.mul);
      EXPECT_EQ(test.b * test.a, test.mul);
      EXPECT_EQ(*(test.a / test.b), test.adb);
      EXPECT_EQ(*(test.b / test.a), test.bda);
    }
  }
}

TEST(ValueTest, MultiplicativeGroupOperators) {
  math::GF7 a = math::GF7::Random();

  struct {
    Value<math::GF7> a;
    Value<math::GF7> inverse;
    Value<math::GF7> sqr;
    Value<math::GF7> pow;
  } tests[] = {
      {
          Value<math::GF7>::Known(a),
          a.IsZero() ? Value<math::GF7>::Unknown()
                     : Value<math::GF7>::Known(*a.Inverse()),
          Value<math::GF7>::Known(a.Square()),
          Value<math::GF7>::Known(a.Pow(5)),
      },
      {
          Value<math::GF7>::Unknown(),
          Value<math::GF7>::Unknown(),
          Value<math::GF7>::Unknown(),
          Value<math::GF7>::Unknown(),
      },
  };

  for (auto& test : tests) {
    if (test.a.IsNone()) {
      EXPECT_TRUE(test.a.Inverse()->IsNone());
      EXPECT_TRUE((*test.a.InverseInPlace())->IsNone());
      EXPECT_TRUE(test.a.Square().IsNone());
      EXPECT_TRUE(test.a.SquareInPlace().IsNone());
      EXPECT_TRUE(test.a.Pow(5).IsNone());
    } else {
      Value<math::GF7> a_tmp;
      if (UNLIKELY(test.a.IsZero())) {
        ASSERT_FALSE(test.a.Inverse());
        ASSERT_FALSE(test.a.InverseInPlace());
      } else {
        EXPECT_EQ(*test.a.Inverse(), test.inverse);
        a_tmp = test.a;
        EXPECT_EQ(**a_tmp.InverseInPlace(), test.inverse);
      }
      EXPECT_EQ(test.a.Square(), test.sqr);
      a_tmp = test.a;
      a_tmp.SquareInPlace();
      EXPECT_EQ(a_tmp, test.sqr);

      EXPECT_EQ(test.a.Pow(5), test.pow);
    }
  }
}

TEST(ValueTest, ToRationalFieldValue) {
  math::GF7 a = math::GF7::Random();
  Value<math::GF7> v = Value<math::GF7>::Known(a);
  EXPECT_EQ(v.ToRationalFieldValue(),
            Value<math::RationalField<math::GF7>>::Known(
                math::RationalField<math::GF7>(a)));

  EXPECT_EQ(Value<math::GF7>::Unknown().ToRationalFieldValue(),
            Value<math::RationalField<math::GF7>>::Unknown());
}

}  // namespace tachyon::zk
