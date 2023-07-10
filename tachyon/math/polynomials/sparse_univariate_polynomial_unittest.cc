#include <optional>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/prime_field.h"
#include "tachyon/math/polynomials/univariate_polynomial.h"

namespace tachyon {
namespace math {

namespace {

const size_t kMaxDegree = 5;

using Poly = SparseUnivariatePolynomial<GF7, kMaxDegree>;
using Coeffs = SparseCoefficients<GF7, kMaxDegree>;

class SparseUnivariatePolynomialTest : public ::testing::Test {
 public:
  SparseUnivariatePolynomialTest() {
    GF7Config::Init();

    polys_.push_back(Poly(Coeffs({{0, GF7(3)}, {2, GF7(1)}, {4, GF7(2)}})));
    polys_.push_back(Poly(Coeffs({{0, GF7(3)}})));
    polys_.push_back(Poly(Coeffs({{3, GF7(5)}})));
    polys_.push_back(Poly(Coeffs({{4, GF7(5)}})));
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
          EXPECT_EQ(*test.poly[i], GF7(test.coefficients[i].value()));
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
    GF7 expected;
  } tests[] = {
      {polys_[0], GF7(6)}, {polys_[1], GF7(3)}, {polys_[2], GF7(2)},
      {polys_[3], GF7(6)}, {polys_[4], GF7(0)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Evaluate(GF7(3)), test.expected);
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
          Poly(Coeffs({{0, GF7(6)}, {2, GF7(1)}, {4, GF7(2)}})),
          Poly(Coeffs({{2, GF7(1)}, {4, GF7(2)}})),
          Poly(Coeffs({{2, GF7(6)}, {4, GF7(5)}})),
      },
      {
          polys_[0],
          polys_[2],
          Poly(Coeffs({{0, GF7(3)}, {2, GF7(1)}, {3, GF7(5)}, {4, GF7(2)}})),
          Poly(Coeffs({{0, GF7(3)}, {2, GF7(1)}, {3, GF7(2)}, {4, GF7(2)}})),
          Poly(Coeffs({{0, GF7(4)}, {2, GF7(6)}, {3, GF7(5)}, {4, GF7(5)}})),
      },
      {
          polys_[0],
          polys_[3],
          Poly(Coeffs({{0, GF7(3)}, {2, GF7(1)}})),
          Poly(Coeffs({{0, GF7(3)}, {2, GF7(1)}, {4, GF7(4)}})),
          Poly(Coeffs({{0, GF7(4)}, {2, GF7(6)}, {4, GF7(3)}})),
      },
      {
          polys_[0],
          polys_[4],
          Poly(Coeffs({{0, GF7(3)}, {2, GF7(1)}, {4, GF7(2)}})),
          Poly(Coeffs({{0, GF7(3)}, {2, GF7(1)}, {4, GF7(2)}})),
          Poly(Coeffs({{0, GF7(4)}, {2, GF7(6)}, {4, GF7(5)}})),
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

}  // namespace math
}  // namespace tachyon
