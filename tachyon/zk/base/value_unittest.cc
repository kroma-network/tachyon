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
      EXPECT_DEATH(test.a + test.b, "");
      EXPECT_DEATH(test.b + test.a, "");
      EXPECT_DEATH(test.a - test.b, "");
      EXPECT_DEATH(test.b - test.a, "");

      Value<math::GF7> tmp = test.a;
      EXPECT_DEATH(tmp += test.b, "");
      EXPECT_DEATH(tmp -= test.b, "");
    } else {
      EXPECT_EQ(test.a + test.b, test.sum);
      EXPECT_EQ(test.b + test.a, test.sum);
      EXPECT_EQ(test.a - test.b, test.amb);
      EXPECT_EQ(test.b - test.a, test.bma);

      Value<math::GF7> tmp = test.a;
      tmp += test.b;
      EXPECT_EQ(tmp, test.sum);
      tmp -= test.b;
      EXPECT_EQ(tmp, test.a);
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
          Value<math::GF7>::Known(a.Negative()),
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
      EXPECT_DEATH(std::ignore = test.a.Negative(), "");
      EXPECT_DEATH(test.a.NegInPlace(), "");
      EXPECT_DEATH(std::ignore = test.a.Double(), "");
      EXPECT_DEATH(test.a.DoubleInPlace(), "");
    } else {
      EXPECT_EQ(test.a.Negative(), test.neg);
      Value<math::GF7> a_tmp = test.a;
      a_tmp.NegInPlace();
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
       Value<math::GF7>::Known(a * b), Value<math::GF7>::Known(a / b),
       Value<math::GF7>::Known(b / a)},
      {Value<math::GF7>::Known(a), Value<math::GF7>::Unknown(),
       Value<math::GF7>::Unknown(), Value<math::GF7>::Unknown(),
       Value<math::GF7>::Unknown()},
  };

  for (const auto& test : tests) {
    if (test.a.IsNone() || test.b.IsNone()) {
      EXPECT_DEATH(test.a * test.b, "");
      EXPECT_DEATH(test.b * test.a, "");
      EXPECT_DEATH(test.a / test.b, "");
      EXPECT_DEATH(test.b / test.a, "");

      Value<math::GF7> tmp = test.a;
      EXPECT_DEATH(tmp *= test.b, "");
      EXPECT_DEATH(tmp /= test.b, "");
    } else {
      EXPECT_EQ(test.a * test.b, test.mul);
      EXPECT_EQ(test.b * test.a, test.mul);
      EXPECT_EQ(test.a / test.b, test.adb);
      EXPECT_EQ(test.b / test.a, test.bda);

      Value<math::GF7> tmp = test.a;
      tmp *= test.b;
      EXPECT_EQ(tmp, test.mul);
      tmp /= test.b;
      EXPECT_EQ(tmp, test.a);
    }
  }
}

TEST(ValueTest, MultiplicativeGroupOperators) {
  math::GF7 a = math::GF7::Random();
  while (a.IsZero()) {
    a = math::GF7::Random();
  }

  struct {
    Value<math::GF7> a;
    Value<math::GF7> inverse;
    Value<math::GF7> sqr;
    Value<math::GF7> pow;
  } tests[] = {
      {
          Value<math::GF7>::Known(a),
          Value<math::GF7>::Known(a.Inverse()),
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
      EXPECT_DEATH(std::ignore = test.a.Inverse(), "");
      EXPECT_DEATH(test.a.InverseInPlace(), "");
      EXPECT_DEATH(std::ignore = test.a.Square(), "");
      EXPECT_DEATH(test.a.SquareInPlace(), "");
      EXPECT_DEATH(std::ignore = test.a.Pow(5), "");
    } else {
      EXPECT_EQ(test.a.Inverse(), test.inverse);
      Value<math::GF7> a_tmp = test.a;
      a_tmp.InverseInPlace();
      EXPECT_EQ(a_tmp, test.inverse);

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
