#include <optional>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/prime_field.h"
#include "tachyon/math/polynomials/univariate_polynomial.h"

namespace tachyon {
namespace math {

namespace {

const size_t kMaxDegree = 5;

using Poly = SparseUnivariatePolynomial<GF7Gmp, kMaxDegree>;
using Coeffs = SparseCoefficients<GF7Gmp, kMaxDegree>;

class SparseUnivariatePolynomialTest : public ::testing::Test {
 public:
  SparseUnivariatePolynomialTest() {
    GF7Config::Init();

    polys_.push_back(
        Poly(Coeffs({{0, GF7Gmp(3)}, {2, GF7Gmp(1)}, {4, GF7Gmp(2)}})));
    polys_.push_back(Poly(Coeffs({{0, GF7Gmp(3)}})));
    polys_.push_back(Poly(Coeffs({{3, GF7Gmp(5)}})));
    polys_.push_back(Poly(Coeffs({{4, GF7Gmp(5)}})));
    polys_.push_back(Poly::Zero());
  }
  SparseUnivariatePolynomialTest(const SparseUnivariatePolynomialTest&) =
      delete;
  SparseUnivariatePolynomialTest& operator=(
      const SparseUnivariatePolynomialTest&) = delete;
  ~SparseUnivariatePolynomialTest() override = default;

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(SparseUnivariatePolynomialTest, IsZero) {
  EXPECT_TRUE(Poly::Zero().IsZero());
  EXPECT_TRUE(Poly(Coeffs({{0, GF7Gmp(0)}})).IsZero());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    EXPECT_FALSE(polys_[i].IsZero());
  }
  EXPECT_TRUE(polys_[polys_.size() - 1].IsZero());
}

TEST_F(SparseUnivariatePolynomialTest, IsOne) {
  EXPECT_TRUE(Poly::One().IsOne());
  EXPECT_TRUE(Poly(Coeffs({{0, GF7Gmp(1)}})).IsOne());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    EXPECT_FALSE(polys_[i].IsOne());
  }
}

TEST_F(SparseUnivariatePolynomialTest, Random) {
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

TEST_F(SparseUnivariatePolynomialTest, IndexingOperator) {
  struct {
    const Poly& poly;
    std::vector<std::optional<int>> coefficients;
  } tests[] = {
      {polys_[0], {3, std::nullopt, 1, std::nullopt, 2}},
      {polys_[1], {3}},
      {polys_[2], {std::nullopt, std::nullopt, std::nullopt, 5}},
      {polys_[3], {std::nullopt, std::nullopt, std::nullopt, std::nullopt, 5}},
      {polys_[4], {}},
  };

  for (const auto& test : tests) {
    for (size_t i = 0; i < kMaxDegree; ++i) {
      if (i < test.coefficients.size()) {
        if (test.coefficients[i].has_value()) {
          EXPECT_EQ(*test.poly[i], GF7Gmp(test.coefficients[i].value()));
        } else {
          EXPECT_EQ(test.poly[i], nullptr);
        }
      } else {
        EXPECT_EQ(test.poly[i], nullptr);
      }
    }
  }
}

TEST_F(SparseUnivariatePolynomialTest, Degree) {
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

TEST_F(SparseUnivariatePolynomialTest, Evaluate) {
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

TEST_F(SparseUnivariatePolynomialTest, ToString) {
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

TEST_F(SparseUnivariatePolynomialTest, AdditiveOperators) {
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
          Poly(Coeffs({{0, GF7Gmp(6)}, {2, GF7Gmp(1)}, {4, GF7Gmp(2)}})),
          Poly(Coeffs({{2, GF7Gmp(1)}, {4, GF7Gmp(2)}})),
          Poly(Coeffs({{2, GF7Gmp(6)}, {4, GF7Gmp(5)}})),
      },
      {
          polys_[0],
          polys_[2],
          Poly(Coeffs({{0, GF7Gmp(3)},
                       {2, GF7Gmp(1)},
                       {3, GF7Gmp(5)},
                       {4, GF7Gmp(2)}})),
          Poly(Coeffs({{0, GF7Gmp(3)},
                       {2, GF7Gmp(1)},
                       {3, GF7Gmp(2)},
                       {4, GF7Gmp(2)}})),
          Poly(Coeffs({{0, GF7Gmp(4)},
                       {2, GF7Gmp(6)},
                       {3, GF7Gmp(5)},
                       {4, GF7Gmp(5)}})),
      },
      {
          polys_[0],
          polys_[3],
          Poly(Coeffs({{0, GF7Gmp(3)}, {2, GF7Gmp(1)}})),
          Poly(Coeffs({{0, GF7Gmp(3)}, {2, GF7Gmp(1)}, {4, GF7Gmp(4)}})),
          Poly(Coeffs({{0, GF7Gmp(4)}, {2, GF7Gmp(6)}, {4, GF7Gmp(3)}})),
      },
      {
          polys_[0],
          polys_[4],
          Poly(Coeffs({{0, GF7Gmp(3)}, {2, GF7Gmp(1)}, {4, GF7Gmp(2)}})),
          Poly(Coeffs({{0, GF7Gmp(3)}, {2, GF7Gmp(1)}, {4, GF7Gmp(2)}})),
          Poly(Coeffs({{0, GF7Gmp(4)}, {2, GF7Gmp(6)}, {4, GF7Gmp(5)}})),
      },
  };

  for (const auto& test : tests) {
    const auto a_dense = test.a.ToDense();
    const auto b_dense = test.b.ToDense();
    const auto sum_dense = test.sum.ToDense();
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a + b_dense, sum_dense);
    EXPECT_EQ(test.b + a_dense, sum_dense);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);
    EXPECT_EQ(test.a - b_dense, test.amb.ToDense());
    EXPECT_EQ(test.b - a_dense, test.bma.ToDense());

    Poly tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(SparseUnivariatePolynomialTest, MultiplicativeOperators) {
  Poly a(Coeffs({{0, GF7Gmp(3)}, {1, GF7Gmp(1)}}));
  Poly b(Coeffs({{0, GF7Gmp(5)}, {1, GF7Gmp(2)}, {2, GF7Gmp(5)}}));
  Poly one = Poly::One();
  Poly zero = Poly::Zero();

  using DensePoly = DenseUnivariatePolynomial<GF7Gmp, kMaxDegree>;
  using DenseCoeffs = DenseCoefficients<GF7Gmp, kMaxDegree>;

  struct {
    const Poly& a;
    const Poly& b;
    Poly mul;
    DensePoly adb;
    DensePoly amb;
    DensePoly bda;
    DensePoly bma;
  } tests[] = {
      {
          a,
          b,
          Poly(Coeffs({{0, GF7Gmp(1)},
                       {1, GF7Gmp(4)},
                       {2, GF7Gmp(3)},
                       {3, GF7Gmp(5)}})),
          zero.ToDense(),
          a.ToDense(),
          DensePoly(DenseCoeffs({GF7Gmp(1), GF7Gmp(5)})),
          DensePoly(DenseCoeffs({GF7Gmp(2)})),
      },
      {
          a,
          one,
          a,
          a.ToDense(),
          zero.ToDense(),
          zero.ToDense(),
          one.ToDense(),
      },
      {
          a,
          zero,
          zero,
          zero.ToDense(),
          zero.ToDense(),
          zero.ToDense(),
          zero.ToDense(),
      },
  };

  for (const auto& test : tests) {
    const auto a_dense = test.a.ToDense();
    const auto b_dense = test.b.ToDense();
    const auto mul_dense = test.mul.ToDense();
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
    EXPECT_EQ(test.a * b_dense, mul_dense);
    EXPECT_EQ(test.b * a_dense, mul_dense);
    if (!b_dense.IsZero()) {
      EXPECT_EQ(test.a / b_dense, test.adb);
      EXPECT_EQ(test.a % b_dense, test.amb);
    }
    if (!a_dense.IsZero()) {
      EXPECT_EQ(test.b / a_dense, test.bda);
      EXPECT_EQ(test.b % a_dense, test.bma);
    }

    {
      Poly tmp = test.a;
      tmp *= test.b;
      EXPECT_EQ(tmp, test.mul);
    }
  }
}

}  // namespace math
}  // namespace tachyon
