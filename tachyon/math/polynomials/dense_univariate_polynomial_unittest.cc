#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/prime_field.h"
#include "tachyon/math/polynomials/univariate_polynomial.h"

namespace tachyon {
namespace math {

namespace {

class DenseUnivariatePolynomialTest : public ::testing::Test {
 public:
  DenseUnivariatePolynomialTest() {
    GF7Config::Init();

    polys_.push_back(DenseUnivariatePolynomial<GF7, 5>(
        DenseCoefficients<GF7, 5>({GF7(3), GF7(0), GF7(1), GF7(0), GF7(2)})));
    polys_.push_back(
        DenseUnivariatePolynomial<GF7, 5>(DenseCoefficients<GF7, 5>({GF7(3)})));
    polys_.push_back(DenseUnivariatePolynomial<GF7, 5>(
        DenseCoefficients<GF7, 5>({GF7(0), GF7(0), GF7(0), GF7(5)})));
  }
  DenseUnivariatePolynomialTest(const DenseUnivariatePolynomialTest&) = delete;
  DenseUnivariatePolynomialTest& operator=(
      const DenseUnivariatePolynomialTest&) = delete;
  ~DenseUnivariatePolynomialTest() override = default;

 protected:
  std::vector<DenseUnivariatePolynomial<GF7, 5>> polys_;
};

}  // namespace

TEST_F(DenseUnivariatePolynomialTest, IndexingOperator) {
  struct {
    const DenseUnivariatePolynomial<GF7, 5>& poly;
    std::vector<int> coefficients;
  } tests[] = {
      {polys_[0], {3, 0, 1, 0, 2}},
      {polys_[1], {3}},
      {polys_[2], {0, 0, 0, 5}},
  };

  for (const auto& test : tests) {
    for (size_t i = 0; i < 5; ++i) {
      if (i < test.coefficients.size()) {
        EXPECT_EQ(*test.poly[i], GF7(test.coefficients[i]));
      } else {
        EXPECT_EQ(test.poly[i], nullptr);
      }
    }
  }
}

TEST_F(DenseUnivariatePolynomialTest, Degree) {
  struct {
    const DenseUnivariatePolynomial<GF7, 5>& poly;
    size_t degree;
  } tests[] = {
      {polys_[0], 4},
      {polys_[1], 0},
      {polys_[2], 3},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Degree(), test.degree);
  }
}

TEST_F(DenseUnivariatePolynomialTest, Evaluate) {
  struct {
    const DenseUnivariatePolynomial<GF7, 5>& poly;
    GF7 expected;
  } tests[] = {
      {polys_[0], GF7(6)},
      {polys_[1], GF7(3)},
      {polys_[2], GF7(2)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Evaluate(GF7(3)), test.expected);
  }
}

TEST_F(DenseUnivariatePolynomialTest, ToString) {
  struct {
    const DenseUnivariatePolynomial<GF7, 5>& poly;
    std::string_view expected;
  } tests[] = {
      {polys_[0], "2 * x^4 + 1 * x^2 + 3"},
      {polys_[1], "3"},
      {polys_[2], "5 * x^3"},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.ToString(), test.expected);
  }
}

}  // namespace math
}  // namespace tachyon
