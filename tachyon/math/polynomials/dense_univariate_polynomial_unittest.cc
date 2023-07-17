#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/prime_field.h"
#include "tachyon/math/polynomials/univariate_polynomial.h"

namespace tachyon {
namespace math {

namespace {

const size_t kMaxDegree = 5;

using Poly = DenseUnivariatePolynomial<GF7Gmp, kMaxDegree>;
using Coeffs = DenseCoefficients<GF7Gmp, kMaxDegree>;

class DenseUnivariatePolynomialTest : public ::testing::Test {
 public:
  DenseUnivariatePolynomialTest() {
    GF7Config::Init();

    polys_.push_back(
        Poly(Coeffs({GF7Gmp(3), GF7Gmp(0), GF7Gmp(1), GF7Gmp(0), GF7Gmp(2)})));
    polys_.push_back(Poly(Coeffs({GF7Gmp(3)})));
    polys_.push_back(
        Poly(Coeffs({GF7Gmp(0), GF7Gmp(0), GF7Gmp(0), GF7Gmp(5)})));
    polys_.push_back(
        Poly(Coeffs({GF7Gmp(0), GF7Gmp(0), GF7Gmp(0), GF7Gmp(0), GF7Gmp(5)})));
    polys_.push_back(Poly::Zero());
  }
  DenseUnivariatePolynomialTest(const DenseUnivariatePolynomialTest&) = delete;
  DenseUnivariatePolynomialTest& operator=(
      const DenseUnivariatePolynomialTest&) = delete;
  ~DenseUnivariatePolynomialTest() override = default;

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(DenseUnivariatePolynomialTest, IsZero) {
  EXPECT_TRUE(Poly::Zero().IsZero());
  EXPECT_TRUE(Poly(Coeffs({GF7Gmp(0)})).IsZero());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    EXPECT_FALSE(polys_[i].IsZero());
  }
  EXPECT_TRUE(polys_[polys_.size() - 1].IsZero());
}

TEST_F(DenseUnivariatePolynomialTest, IsOne) {
  EXPECT_TRUE(Poly::One().IsOne());
  EXPECT_TRUE(Poly(Coeffs({GF7Gmp(1)})).IsOne());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    EXPECT_FALSE(polys_[i].IsOne());
  }
}

TEST_F(DenseUnivariatePolynomialTest, Random) {
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

TEST_F(DenseUnivariatePolynomialTest, IndexingOperator) {
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
        EXPECT_EQ(*test.poly[i], GF7Gmp(test.coefficients[i]));
      } else {
        EXPECT_EQ(test.poly[i], nullptr);
      }
    }
  }
}

TEST_F(DenseUnivariatePolynomialTest, Degree) {
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
}

TEST_F(DenseUnivariatePolynomialTest, Evaluate) {
  struct {
    const Poly& poly;
    GF7Gmp expected;
  } tests[] = {
      {polys_[0], GF7Gmp(6)}, {polys_[1], GF7Gmp(3)}, {polys_[2], GF7Gmp(2)},
      {polys_[3], GF7Gmp(6)}, {polys_[4], GF7Gmp(0)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Evaluate(GF7Gmp(3)), test.expected);
  }
}

TEST_F(DenseUnivariatePolynomialTest, ToString) {
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

TEST_F(DenseUnivariatePolynomialTest, AdditiveOperators) {
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
          Poly(Coeffs({GF7Gmp(6), GF7Gmp(0), GF7Gmp(1), GF7Gmp(0), GF7Gmp(2)})),
          Poly(Coeffs({GF7Gmp(0), GF7Gmp(0), GF7Gmp(1), GF7Gmp(0), GF7Gmp(2)})),
          Poly(Coeffs({GF7Gmp(0), GF7Gmp(0), GF7Gmp(6), GF7Gmp(0), GF7Gmp(5)})),
      },
      {
          polys_[0],
          polys_[2],
          Poly(Coeffs({GF7Gmp(3), GF7Gmp(0), GF7Gmp(1), GF7Gmp(5), GF7Gmp(2)})),
          Poly(Coeffs({GF7Gmp(3), GF7Gmp(0), GF7Gmp(1), GF7Gmp(2), GF7Gmp(2)})),
          Poly(Coeffs({GF7Gmp(4), GF7Gmp(0), GF7Gmp(6), GF7Gmp(5), GF7Gmp(5)})),
      },
      {
          polys_[0],
          polys_[3],
          Poly(Coeffs({GF7Gmp(3), GF7Gmp(0), GF7Gmp(1)})),
          Poly(Coeffs({GF7Gmp(3), GF7Gmp(0), GF7Gmp(1), GF7Gmp(0), GF7Gmp(4)})),
          Poly(Coeffs({GF7Gmp(4), GF7Gmp(0), GF7Gmp(6), GF7Gmp(0), GF7Gmp(3)})),
      },
      {
          polys_[0],
          polys_[4],
          Poly(Coeffs({GF7Gmp(3), GF7Gmp(0), GF7Gmp(1), GF7Gmp(0), GF7Gmp(2)})),
          Poly(Coeffs({GF7Gmp(3), GF7Gmp(0), GF7Gmp(1), GF7Gmp(0), GF7Gmp(2)})),
          Poly(Coeffs({GF7Gmp(4), GF7Gmp(0), GF7Gmp(6), GF7Gmp(0), GF7Gmp(5)})),
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

TEST_F(DenseUnivariatePolynomialTest, MultiplicativeOperators) {
  Poly a(Coeffs({GF7Gmp(3), GF7Gmp(1)}));
  Poly b(Coeffs({GF7Gmp(5), GF7Gmp(2), GF7Gmp(5)}));
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
          Poly(Coeffs({GF7Gmp(1), GF7Gmp(4), GF7Gmp(3), GF7Gmp(5)})),
          zero,
          a,
          Poly(Coeffs({GF7Gmp(1), GF7Gmp(5)})),
          Poly(Coeffs({GF7Gmp(2)})),
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

    {
      Poly tmp = test.a;
      tmp *= test.b;
      EXPECT_EQ(tmp, test.mul);
      if (!test.b.IsZero()) {
        tmp /= test.b;
        EXPECT_EQ(tmp, test.a);
      }
    }
  }
}

}  // namespace math
}  // namespace tachyon
