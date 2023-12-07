#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

namespace {

const size_t kMaxDegree = 5;

using Poly = UnivariateDensePolynomial<GF7, kMaxDegree>;
using Coeffs = UnivariateDenseCoefficients<GF7, kMaxDegree>;

class UnivariateDensePolynomialTest : public testing::Test {
 public:
  static void SetUpTestSuite() { GF7::Init(); }

  void SetUp() override {
    polys_.push_back(Poly(Coeffs({GF7(3), GF7(0), GF7(1), GF7(0), GF7(2)})));
    polys_.push_back(Poly(Coeffs({GF7(3)})));
    polys_.push_back(Poly(Coeffs({GF7(0), GF7(0), GF7(0), GF7(5)})));
    polys_.push_back(Poly(Coeffs({GF7(0), GF7(0), GF7(0), GF7(0), GF7(5)})));
    polys_.push_back(Poly::Zero());
  }

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(UnivariateDensePolynomialTest, IsZero) {
  EXPECT_TRUE(Poly().IsZero());
  EXPECT_TRUE(Poly::Zero().IsZero());
  EXPECT_TRUE(Poly(Coeffs({GF7(0)})).IsZero());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    EXPECT_FALSE(polys_[i].IsZero());
  }
  EXPECT_TRUE(polys_[polys_.size() - 1].IsZero());
}

TEST_F(UnivariateDensePolynomialTest, IsOne) {
  EXPECT_TRUE(Poly::One().IsOne());
  EXPECT_TRUE(Poly(Coeffs({GF7(1)})).IsOne());
  for (size_t i = 0; i < polys_.size(); ++i) {
    EXPECT_FALSE(polys_[i].IsOne());
  }
}

TEST_F(UnivariateDensePolynomialTest, Random) {
  bool success = false;
  Poly r = Poly::Random(kMaxDegree);
  for (size_t i = 0; i < 100; ++i) {
    if (r != Poly::Random(kMaxDegree)) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(UnivariateDensePolynomialTest, Linearize) {
  std::vector<Poly> polys = {
      Poly::Random(kMaxDegree),
      Poly::Random(kMaxDegree),
      Poly::Random(kMaxDegree),
  };
  GF7 y = GF7::Random();
  Poly linearized = Poly::Linearize(polys, y);

  GF7 x = GF7::Random();
  GF7 expected = polys[0].Evaluate(x);
  GF7 power = y;
  for (size_t i = 1; i < polys.size(); ++i) {
    expected += polys[i].Evaluate(x) * power;
    power *= y;
  }

  EXPECT_EQ(expected, linearized.Evaluate(x));

  Poly linearized2 = Poly::LinearizeInPlace(polys, y);
  EXPECT_EQ(linearized, linearized2);
}

TEST_F(UnivariateDensePolynomialTest, IndexingOperator) {
  struct {
    const Poly& poly;
    std::vector<int> coefficients;
  } tests[] = {
      {polys_[0], {3, 0, 1, 0, 2}}, {polys_[1], {3}}, {polys_[2], {0, 0, 0, 5}},
      {polys_[3], {0, 0, 0, 0, 5}}, {polys_[4], {}},
  };

  for (const auto& test : tests) {
    for (size_t i = 0; i < kMaxDegree; ++i) {
      if (i < test.coefficients.size()) {
        EXPECT_EQ(*test.poly[i], GF7(test.coefficients[i]));
      } else {
        EXPECT_EQ(test.poly[i], nullptr);
      }
    }
  }
}

TEST_F(UnivariateDensePolynomialTest, Degree) {
  struct {
    const Poly& poly;
    size_t degree;
  } tests[] = {
      {polys_[0], 4}, {polys_[1], 0}, {polys_[2], 3},
      {polys_[3], 4}, {polys_[4], 0},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Degree(), test.degree);
  }
  EXPECT_LE(Poly::Random(kMaxDegree).Degree(), kMaxDegree);
}

TEST_F(UnivariateDensePolynomialTest, Evaluate) {
  struct {
    const Poly& poly;
    GF7 expected;
  } tests[] = {
      {polys_[0], GF7(6)}, {polys_[1], GF7(3)}, {polys_[2], GF7(2)},
      {polys_[3], GF7(6)}, {polys_[4], GF7(0)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Evaluate(GF7(3)), test.expected);
  }
}

TEST_F(UnivariateDensePolynomialTest, ToString) {
  struct {
    const Poly& poly;
    std::string_view expected;
  } tests[] = {
      {polys_[0], "2 * x^4 + 1 * x^2 + 3"},
      {polys_[1], "3"},
      {polys_[2], "5 * x^3"},
      {polys_[3], "5 * x^4"},
      {polys_[4], ""},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.ToString(), test.expected);
  }
}

TEST_F(UnivariateDensePolynomialTest, AdditiveOperators) {
  struct {
    const Poly& a;
    const Poly& b;
    Poly sum;
    Poly amb;
    Poly bma;
  } tests[] = {
      {
          polys_[0],
          polys_[1],
          Poly(Coeffs({GF7(6), GF7(0), GF7(1), GF7(0), GF7(2)})),
          Poly(Coeffs({GF7(0), GF7(0), GF7(1), GF7(0), GF7(2)})),
          Poly(Coeffs({GF7(0), GF7(0), GF7(6), GF7(0), GF7(5)})),
      },
      {
          polys_[0],
          polys_[2],
          Poly(Coeffs({GF7(3), GF7(0), GF7(1), GF7(5), GF7(2)})),
          Poly(Coeffs({GF7(3), GF7(0), GF7(1), GF7(2), GF7(2)})),
          Poly(Coeffs({GF7(4), GF7(0), GF7(6), GF7(5), GF7(5)})),
      },
      {
          polys_[0],
          polys_[3],
          Poly(Coeffs({GF7(3), GF7(0), GF7(1)})),
          Poly(Coeffs({GF7(3), GF7(0), GF7(1), GF7(0), GF7(4)})),
          Poly(Coeffs({GF7(4), GF7(0), GF7(6), GF7(0), GF7(3)})),
      },
      {
          polys_[0],
          polys_[4],
          Poly(Coeffs({GF7(3), GF7(0), GF7(1), GF7(0), GF7(2)})),
          Poly(Coeffs({GF7(3), GF7(0), GF7(1), GF7(0), GF7(2)})),
          Poly(Coeffs({GF7(4), GF7(0), GF7(6), GF7(0), GF7(5)})),
      },
  };

  for (const auto& test : tests) {
    const auto a_sparse = test.a.ToSparse();
    const auto b_sparse = test.b.ToSparse();
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a + b_sparse, test.sum);
    EXPECT_EQ(test.b + a_sparse, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);
    EXPECT_EQ(test.a - b_sparse, test.amb);
    EXPECT_EQ(test.b - a_sparse, test.bma);

    {
      Poly tmp = test.a;
      tmp += test.b;
      EXPECT_EQ(tmp, test.sum);
      tmp -= test.b;
      EXPECT_EQ(tmp, test.a);
    }
    {
      Poly tmp = test.a;
      tmp += b_sparse;
      EXPECT_EQ(tmp, test.sum);
      tmp -= b_sparse;
      EXPECT_EQ(tmp, test.a);
    }
  }
}

TEST_F(UnivariateDensePolynomialTest, MultiplicativeOperators) {
  Poly a(Coeffs({GF7(3), GF7(1)}));
  Poly b(Coeffs({GF7(5), GF7(2), GF7(5)}));
  Poly one = Poly::One();
  Poly zero = Poly::Zero();

  struct {
    const Poly& a;
    const Poly& b;
    Poly mul;
    Poly adb;
    Poly amb;
    Poly bda;
    Poly bma;
  } tests[] = {
      {
          a,
          b,
          Poly(Coeffs({GF7(1), GF7(4), GF7(3), GF7(5)})),
          zero,
          a,
          Poly(Coeffs({GF7(1), GF7(5)})),
          Poly(Coeffs({GF7(2)})),
      },
      {
          a,
          one,
          a,
          a,
          zero,
          zero,
          one,
      },
      {
          a,
          zero,
          zero,
          zero,
          zero,
          zero,
          zero,
      },
  };

  for (const auto& test : tests) {
    const auto a_sparse = test.a.ToSparse();
    const auto b_sparse = test.b.ToSparse();
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    if (!test.b.IsZero()) {
      EXPECT_EQ(test.a / test.b, test.adb);
      EXPECT_EQ(test.a % test.b, test.amb);
    }
    if (!test.a.IsZero()) {
      EXPECT_EQ(test.b / test.a, test.bda);
      EXPECT_EQ(test.b % test.a, test.bma);
    }
    EXPECT_EQ(test.a * b_sparse, test.mul);
    EXPECT_EQ(test.b * a_sparse, test.mul);
    if (!b_sparse.IsZero()) {
      EXPECT_EQ(test.a / b_sparse, test.adb);
      EXPECT_EQ(test.a % b_sparse, test.amb);
    }
    if (!a_sparse.IsZero()) {
      EXPECT_EQ(test.b / a_sparse, test.bda);
      EXPECT_EQ(test.b % a_sparse, test.bma);
    }

    Poly tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    if (!test.b.IsZero()) {
      tmp /= test.b;
      EXPECT_EQ(tmp, test.a);
    }
  }
}

TEST_F(UnivariateDensePolynomialTest, MulScalar) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 scalar = GF7::Random();

  std::vector<GF7> expected_coeffs;
  const std::vector<GF7>& coeffs = poly.coefficients().coefficients();
  expected_coeffs.reserve(coeffs.size());
  for (size_t i = 0; i < coeffs.size(); ++i) {
    expected_coeffs.push_back(coeffs[i] * scalar);
  }

  Poly actual = poly * scalar;
  Poly expected(Coeffs(std::move(expected_coeffs)));
  EXPECT_EQ(actual, expected);
  poly *= scalar;
  EXPECT_EQ(poly, expected);
}

TEST_F(UnivariateDensePolynomialTest, DivScalar) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 scalar = GF7::Random();
  while (scalar.IsZero()) {
    scalar = GF7::Random();
  }

  std::vector<GF7> expected_coeffs;
  const std::vector<GF7>& coeffs = poly.coefficients().coefficients();
  expected_coeffs.reserve(coeffs.size());
  for (size_t i = 0; i < coeffs.size(); ++i) {
    expected_coeffs.push_back(coeffs[i] / scalar);
  }

  Poly actual = poly / scalar;
  Poly expected(Coeffs(std::move(expected_coeffs)));
  EXPECT_EQ(actual, expected);
  poly /= scalar;
  EXPECT_EQ(poly, expected);
}

TEST_F(UnivariateDensePolynomialTest, FromRoots) {
  // poly = x⁴ + 2x² + 4 = (x - 1)(x - 2)(x + 1)(x + 2)
  Poly poly = Poly(Coeffs({GF7(4), GF7::Zero(), GF7(2), GF7::Zero(), GF7(1)}));
  std::vector<GF7> roots = {GF7(1), GF7(2), GF7(6), GF7(5)};
  EXPECT_EQ(Poly::FromRoots(roots), poly);
}

TEST_F(UnivariateDensePolynomialTest, Copyable) {
  Poly expected(Coeffs({GF7(1), GF7(4), GF7(3), GF7(5)}));
  Poly value;

  base::VectorBuffer buf;
  buf.Write(expected);

  buf.set_buffer_offset(0);
  buf.Read(&value);

  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::math
