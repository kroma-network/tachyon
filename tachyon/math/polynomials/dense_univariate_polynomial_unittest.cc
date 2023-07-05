#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/prime_field.h"
#include "tachyon/math/polynomials/univariate_polynomial.h"

namespace tachyon {
namespace math {

namespace {

class DenseUnivariatePolynomialTest : public ::testing::Test {
 public:
  DenseUnivariatePolynomialTest() {
    Fp7::Init();

    polys_.push_back(DenseUnivariatePolynomial<Fp7, 5>(
        DenseCoefficients<Fp7, 5>({Fp7(3), Fp7(0), Fp7(1), Fp7(0), Fp7(2)})));
    polys_.push_back(
        DenseUnivariatePolynomial<Fp7, 5>(DenseCoefficients<Fp7, 5>({Fp7(3)})));
    polys_.push_back(DenseUnivariatePolynomial<Fp7, 5>(
        DenseCoefficients<Fp7, 5>({Fp7(0), Fp7(0), Fp7(0), Fp7(5)})));
  }
  DenseUnivariatePolynomialTest(const DenseUnivariatePolynomialTest&) = delete;
  DenseUnivariatePolynomialTest& operator=(
      const DenseUnivariatePolynomialTest&) = delete;
  ~DenseUnivariatePolynomialTest() override = default;

 protected:
  std::vector<DenseUnivariatePolynomial<Fp7, 5>> polys_;
};

}  // namespace

TEST_F(DenseUnivariatePolynomialTest, IndexingOperator) {
  struct {
    const DenseUnivariatePolynomial<Fp7, 5>& poly;
    std::vector<int> coefficients;
  } tests[] = {
      {polys_[0], {3, 0, 1, 0, 2}},
      {polys_[1], {3}},
      {polys_[2], {0, 0, 0, 5}},
  };

  for (const auto& test : tests) {
    for (size_t i = 0; i < 5; ++i) {
      if (i < test.coefficients.size()) {
        EXPECT_EQ(*test.poly[i], Fp7(test.coefficients[i]));
      } else {
        EXPECT_EQ(test.poly[i], nullptr);
      }
    }
  }
}

TEST_F(DenseUnivariatePolynomialTest, Degree) {
  struct {
    const DenseUnivariatePolynomial<Fp7, 5>& poly;
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
    const DenseUnivariatePolynomial<Fp7, 5>& poly;
    Fp7 expected;
  } tests[] = {
      {polys_[0], Fp7(6)},
      {polys_[1], Fp7(3)},
      {polys_[2], Fp7(2)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Evaluate(Fp7(3)), test.expected);
  }
}

TEST_F(DenseUnivariatePolynomialTest, ToString) {
  struct {
    const DenseUnivariatePolynomial<Fp7, 5>& poly;
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
