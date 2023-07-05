#include <optional>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/prime_field.h"
#include "tachyon/math/polynomials/univariate_polynomial.h"

namespace tachyon {
namespace math {

namespace {

class SparseUnivariatePolynomialTest : public ::testing::Test {
 public:
  SparseUnivariatePolynomialTest() {
    Fp7::Init();

    polys_.push_back(SparseUnivariatePolynomial<Fp7, 5>(
        SparseCoefficients<Fp7, 5>({{0, Fp7(3)}, {2, Fp7(1)}, {4, Fp7(2)}})));
    polys_.push_back(SparseUnivariatePolynomial<Fp7, 5>(
        SparseCoefficients<Fp7, 5>({{0, Fp7(3)}})));
    polys_.push_back(SparseUnivariatePolynomial<Fp7, 5>(
        SparseCoefficients<Fp7, 5>({{3, Fp7(5)}})));
  }
  SparseUnivariatePolynomialTest(const SparseUnivariatePolynomialTest&) =
      delete;
  SparseUnivariatePolynomialTest& operator=(
      const SparseUnivariatePolynomialTest&) = delete;
  ~SparseUnivariatePolynomialTest() override = default;

 protected:
  std::vector<SparseUnivariatePolynomial<Fp7, 5>> polys_;
};

}  // namespace

TEST_F(SparseUnivariatePolynomialTest, IndexingOperator) {
  struct {
    const SparseUnivariatePolynomial<Fp7, 5>& poly;
    std::vector<std::optional<int>> coefficients;
  } tests[] = {
      {polys_[0], {3, std::nullopt, 1, std::nullopt, 2}},
      {polys_[1], {3}},
      {polys_[2], {std::nullopt, std::nullopt, std::nullopt, 5}},
  };

  for (const auto& test : tests) {
    for (size_t i = 0; i < 5; ++i) {
      if (i < test.coefficients.size()) {
        if (test.coefficients[i].has_value()) {
          EXPECT_EQ(*test.poly[i], Fp7(test.coefficients[i].value()));
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
    const SparseUnivariatePolynomial<Fp7, 5>& poly;
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

TEST_F(SparseUnivariatePolynomialTest, Evaluate) {
  struct {
    const SparseUnivariatePolynomial<Fp7, 5>& poly;
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

TEST_F(SparseUnivariatePolynomialTest, ToString) {
  struct {
    const SparseUnivariatePolynomial<Fp7, 5>& poly;
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
