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
    GF7Config::Init();

    polys_.push_back(SparseUnivariatePolynomial<GF7, 5>(
        SparseCoefficients<GF7, 5>({{0, GF7(3)}, {2, GF7(1)}, {4, GF7(2)}})));
    polys_.push_back(SparseUnivariatePolynomial<GF7, 5>(
        SparseCoefficients<GF7, 5>({{0, GF7(3)}})));
    polys_.push_back(SparseUnivariatePolynomial<GF7, 5>(
        SparseCoefficients<GF7, 5>({{3, GF7(5)}})));
  }
  SparseUnivariatePolynomialTest(const SparseUnivariatePolynomialTest&) =
      delete;
  SparseUnivariatePolynomialTest& operator=(
      const SparseUnivariatePolynomialTest&) = delete;
  ~SparseUnivariatePolynomialTest() override = default;

 protected:
  std::vector<SparseUnivariatePolynomial<GF7, 5>> polys_;
};

}  // namespace

TEST_F(SparseUnivariatePolynomialTest, IndexingOperator) {
  struct {
    const SparseUnivariatePolynomial<GF7, 5>& poly;
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
    const SparseUnivariatePolynomial<GF7, 5>& poly;
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
    const SparseUnivariatePolynomial<GF7, 5>& poly;
    GF7 expectd;
  } tests[] = {
      {polys_[0], GF7(6)},
      {polys_[1], GF7(3)},
      {polys_[2], GF7(2)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Evaluate(GF7(3)), test.expectd);
  }
}

TEST_F(SparseUnivariatePolynomialTest, ToString) {
  struct {
    const SparseUnivariatePolynomial<GF7, 5>& poly;
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
