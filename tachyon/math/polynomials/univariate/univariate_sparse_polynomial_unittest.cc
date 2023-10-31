#include <optional>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

namespace {

constexpr size_t kMaxSize = 6;

using Poly = UnivariateSparsePolynomial<GF7, kMaxSize>;
using Coeffs = UnivariateSparseCoefficients<GF7, kMaxSize>;

class UnivariateSparsePolynomialTest : public testing::Test {
 public:
  static void SetUpTestSuite() { GF7::Init(); }

  void SetUp() override {
    polys_.push_back(Poly(Coeffs({{0, GF7(3)}, {2, GF7(1)}, {4, GF7(2)}})));
    polys_.push_back(Poly(Coeffs({{0, GF7(3)}})));
    polys_.push_back(Poly(Coeffs({{3, GF7(5)}})));
    polys_.push_back(Poly(Coeffs({{4, GF7(5)}})));
    polys_.push_back(Poly(Coeffs({{0, GF7(3)}, {1, GF7(4)}})));
    polys_.push_back(Poly(Coeffs({{0, GF7(3)}, {1, GF7(4)}, {2, GF7(1)}})));
    polys_.push_back(Poly::Zero());
  }

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(UnivariateSparsePolynomialTest, IsZero) {
  EXPECT_TRUE(Poly().IsZero());
  EXPECT_TRUE(Poly::Zero().IsZero());
  EXPECT_TRUE(Poly(Coeffs({{0, GF7(0)}})).IsZero());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    EXPECT_FALSE(polys_[i].IsZero());
  }
  EXPECT_TRUE(polys_[polys_.size() - 1].IsZero());
}

TEST_F(UnivariateSparsePolynomialTest, IsOne) {
  EXPECT_TRUE(Poly::One().IsOne());
  EXPECT_TRUE(Poly(Coeffs({{0, GF7(1)}})).IsOne());
  EXPECT_FALSE(Poly(Coeffs({{1, GF7(1)}})).IsOne());
  for (size_t i = 0; i < polys_.size(); ++i) {
    EXPECT_FALSE(polys_[i].IsOne());
  }
}

TEST_F(UnivariateSparsePolynomialTest, Random) {
  bool success = false;
  Poly r = Poly::Random(kMaxSize);
  for (size_t i = 0; i < 100; ++i) {
    if (r != Poly::Random(kMaxSize)) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(UnivariateSparsePolynomialTest, IndexingOperator) {
  struct {
    const Poly& poly;
    std::vector<std::optional<int>> coefficients;
  } tests[] = {
      {polys_[0], {3, std::nullopt, 1, std::nullopt, 2}},
      {polys_[1], {3}},
      {polys_[2], {std::nullopt, std::nullopt, std::nullopt, 5}},
      {polys_[3], {std::nullopt, std::nullopt, std::nullopt, std::nullopt, 5}},
      {polys_[4], {3, 4}},
      {polys_[5], {3, 4, 1}},
      {polys_[6], {}},
  };

  for (const auto& test : tests) {
    for (size_t i = 0; i < kMaxSize; ++i) {
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

TEST_F(UnivariateSparsePolynomialTest, Degree) {
  struct {
    const Poly& poly;
    size_t degree;
  } tests[] = {
      {polys_[0], 4}, {polys_[1], 0}, {polys_[2], 3}, {polys_[3], 4},
      {polys_[4], 1}, {polys_[5], 2}, {polys_[6], 0},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Degree(), test.degree);
  }
  EXPECT_LE(Poly::Random(kMaxSize).Degree(), kMaxSize - 1);
}

TEST_F(UnivariateSparsePolynomialTest, Evaluate) {
  struct {
    const Poly& poly;
    GF7 expected;
  } tests[] = {
      {polys_[0], GF7(6)}, {polys_[1], GF7(3)}, {polys_[2], GF7(2)},
      {polys_[3], GF7(6)}, {polys_[4], GF7(1)}, {polys_[5], GF7(3)},
      {polys_[6], GF7(0)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Evaluate(GF7(3)), test.expected);
  }
}

TEST_F(UnivariateSparsePolynomialTest, ToString) {
  struct {
    const Poly& poly;
    std::string_view expected;
  } tests[] = {
      {polys_[0], "2 * x^4 + 1 * x^2 + 3"},
      {polys_[1], "3"},
      {polys_[2], "5 * x^3"},
      {polys_[3], "5 * x^4"},
      {polys_[4], "4 * x + 3"},
      {polys_[5], "1 * x^2 + 4 * x + 3"},
      {polys_[6], ""},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.ToString(), test.expected);
  }
}

TEST_F(UnivariateSparsePolynomialTest, AdditiveOperators) {
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
          Poly(Coeffs({{0, GF7(6)}, {1, GF7(4)}, {2, GF7(1)}, {4, GF7(2)}})),
          Poly(Coeffs({{1, GF7(3)}, {2, GF7(1)}, {4, GF7(2)}})),
          Poly(Coeffs({{1, GF7(4)}, {2, GF7(6)}, {4, GF7(5)}})),
      },
      {
          polys_[0],
          polys_[5],
          Poly(Coeffs({{0, GF7(6)}, {1, GF7(4)}, {2, GF7(2)}, {4, GF7(2)}})),
          Poly(Coeffs({{1, GF7(3)}, {4, GF7(2)}})),
          Poly(Coeffs({{1, GF7(4)}, {4, GF7(5)}})),
      },
      {
          polys_[0],
          polys_[6],
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

TEST_F(UnivariateSparsePolynomialTest, MultiplicativeOperators) {
  Poly a(Coeffs({{0, GF7(3)}, {1, GF7(1)}}));
  Poly b(Coeffs({{0, GF7(5)}, {1, GF7(2)}, {2, GF7(5)}}));
  Poly one = Poly::One();
  Poly zero = Poly::Zero();

  using DensePoly = UnivariateDensePolynomial<GF7, kMaxSize>;
  using DenseCoeffs = UnivariateDenseCoefficients<GF7, kMaxSize>;

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
          Poly(Coeffs({{0, GF7(1)}, {1, GF7(4)}, {2, GF7(3)}, {3, GF7(5)}})),
          zero.ToDense(),
          a.ToDense(),
          DensePoly(DenseCoeffs({GF7(1), GF7(5)})),
          DensePoly(DenseCoeffs({GF7(2)})),
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

    Poly tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
  }
}

TEST_F(UnivariateSparsePolynomialTest, Copyable) {
  Poly expected(Coeffs({{0, GF7(3)}, {1, GF7(1)}}));
  Poly value;

  base::VectorBuffer buf;
  buf.Write(expected);

  buf.set_buffer_offset(0);
  buf.Read(&value);

  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::math
