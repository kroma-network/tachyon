#include <optional>

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/base/containers/cxx20_erase_vector.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

namespace {

const size_t kMaxDegree = 5;

using Poly = UnivariateSparsePolynomial<GF7, kMaxDegree>;
using Coeffs = UnivariateSparseCoefficients<GF7, kMaxDegree>;

class UnivariateSparsePolynomialTest : public FiniteFieldTest<GF7> {
 public:
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
  EXPECT_TRUE(Poly(Coeffs({{0, GF7(0)}, {1, GF7(0)}}, true)).IsZero());

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
  Poly r = Poly::Random(kMaxDegree);
  for (size_t i = 0; i < 100; ++i) {
    if (r != Poly::Random(kMaxDegree)) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(UnivariateSparsePolynomialTest, CreateChecked) {
  EXPECT_DEATH(Coeffs::CreateChecked({{2, GF7(3)}, {0, GF7(1)}, {4, GF7(2)}}),
               "");
  EXPECT_EQ(
      Poly(Coeffs::CreateChecked({{0, GF7(3)}, {2, GF7(1)}, {4, GF7(2)}})),
      polys_[0]);
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
    for (size_t i = 0; i < kMaxDegree; ++i) {
      if (i < test.coefficients.size()) {
        if (test.coefficients[i].has_value()) {
          EXPECT_EQ(test.poly[i], GF7(test.coefficients[i].value()));
        } else {
          EXPECT_EQ(test.poly[i], GF7::Zero());
        }
      } else {
        EXPECT_EQ(test.poly[i], GF7::Zero());
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
  EXPECT_LE(Poly::Random(kMaxDegree).Degree(), kMaxDegree);
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

  using DensePoly = UnivariateDensePolynomial<GF7, kMaxDegree>;
  using DenseCoeffs = UnivariateDenseCoefficients<GF7, kMaxDegree>;

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
          a.ToDense(),
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
    } else {
      ASSERT_FALSE(test.a / test.b);
    }
    if (!test.a.IsZero()) {
      EXPECT_EQ(test.b / test.a, test.bda);
      EXPECT_EQ(test.b % test.a, test.bma);
    } else {
      ASSERT_FALSE(test.b / test.a);
    }
    EXPECT_EQ(test.a * b_dense, mul_dense);
    EXPECT_EQ(test.b * a_dense, mul_dense);
    if (!b_dense.IsZero()) {
      EXPECT_EQ(test.a / b_dense, test.adb);
      EXPECT_EQ(test.a % b_dense, test.amb);
    } else {
      ASSERT_FALSE(test.a / b_dense);
    }
    if (!a_dense.IsZero()) {
      EXPECT_EQ(test.b / a_dense, test.bda);
      EXPECT_EQ(test.b % a_dense, test.bma);
    } else {
      ASSERT_FALSE(test.a / a_dense);
    }

    Poly tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
  }
}

TEST_F(UnivariateSparsePolynomialTest, MulScalar) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 scalar = GF7::Random();

  std::vector<UnivariateTerm<GF7>> expected_terms;
  const std::vector<UnivariateTerm<GF7>>& terms = poly.coefficients().terms();
  expected_terms.reserve(terms.size());
  for (size_t i = 0; i < terms.size(); ++i) {
    expected_terms.push_back(terms[i] * scalar);
  }

  Poly actual = poly * scalar;
  Poly expected(Coeffs(std::move(expected_terms), true));
  EXPECT_EQ(actual, expected);
  poly *= scalar;
  EXPECT_EQ(poly, expected);
}

TEST_F(UnivariateSparsePolynomialTest, DivScalar) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 scalar = GF7::Random();
  // NOTE(ashjeong): |scalar| is guaranteed to not be zero
  while (scalar.IsZero()) {
    scalar = GF7::Random();
  }

  std::vector<UnivariateTerm<GF7>> expected_terms;
  const std::vector<UnivariateTerm<GF7>>& terms = poly.coefficients().terms();
  expected_terms.reserve(terms.size());
  for (size_t i = 0; i < terms.size(); ++i) {
    expected_terms.push_back(terms[i] / scalar);
  }

  Poly actual = unwrap<Poly>(poly / scalar);
  Poly expected(Coeffs(std::move(expected_terms)));
  EXPECT_EQ(actual, expected);
  ASSERT_TRUE(poly /= scalar);
  EXPECT_EQ(poly, expected);
}

TEST_F(UnivariateSparsePolynomialTest, FromRoots) {
  // poly = x⁴ + 2x² + 4 = (x - 1)(x - 2)(x + 1)(x + 2)
  Poly poly = Poly(Coeffs({{0, GF7(4)}, {2, GF7(2)}, {4, GF7(1)}}));
  std::vector<GF7> roots = {GF7(1), GF7(2), GF7(6), GF7(5)};
  EXPECT_EQ(Poly::FromRoots(roots), poly);
}

TEST_F(UnivariateSparsePolynomialTest, EvaluateVanishingPolyByRoots) {
  // poly = x⁴ + 2x² + 4 = (x - 1)(x - 2)(x + 1)(x + 2)
  Poly poly = Poly(Coeffs({{0, GF7(4)}, {2, GF7(2)}, {4, GF7(1)}}));
  std::vector<GF7> roots = {GF7(1), GF7(2), GF7(6), GF7(5)};
  GF7 point = GF7::Random();
  EXPECT_EQ(Poly::EvaluateVanishingPolyByRoots(roots, point),
            poly.Evaluate(point));
}

TEST_F(UnivariateSparsePolynomialTest, FoldEven) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 r = GF7::Random();
  Poly folded = poly.Fold<true>(r);
  std::vector<UnivariateTerm<GF7>> terms{{0, r * poly[0] + poly[1]},
                                         {1, r * poly[2] + poly[3]},
                                         {2, r * poly[4] + poly[5]}};
  base::EraseIf(terms, [](const UnivariateTerm<GF7>& term) {
    return term.coefficient.IsZero();
  });
  EXPECT_EQ(folded, Poly(Coeffs(std::move(terms))));

  GF7 r2 = GF7::Random();
  Poly folded2 = folded.Fold<true>(r2);
  terms = {{0, r2 * folded[0] + folded[1]}, {1, r2 * folded[2]}};
  base::EraseIf(terms, [](const UnivariateTerm<GF7>& term) {
    return term.coefficient.IsZero();
  });
  EXPECT_EQ(folded2, Poly(Coeffs(std::move(terms))));

  GF7 r3 = GF7::Random();
  Poly folded3 = folded2.Fold<true>(r3);
  terms = {{0, r3 * folded2[0] + folded2[1]}};
  base::EraseIf(terms, [](const UnivariateTerm<GF7>& term) {
    return term.coefficient.IsZero();
  });
  EXPECT_EQ(folded3, Poly(Coeffs(std::move(terms))));
}

TEST_F(UnivariateSparsePolynomialTest, FoldOdd) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 r = GF7::Random();
  Poly folded = poly.Fold<false>(r);
  std::vector<UnivariateTerm<GF7>> terms{{0, poly[0] + r * poly[1]},
                                         {1, poly[2] + r * poly[3]},
                                         {2, poly[4] + r * poly[5]}};
  base::EraseIf(terms, [](const UnivariateTerm<GF7>& term) {
    return term.coefficient.IsZero();
  });
  EXPECT_EQ(folded, Poly(Coeffs(std::move(terms))));

  GF7 r2 = GF7::Random();
  Poly folded2 = folded.Fold<false>(r2);
  terms = {{0, folded[0] + r2 * folded[1]}, {1, folded[2]}};
  base::EraseIf(terms, [](const UnivariateTerm<GF7>& term) {
    return term.coefficient.IsZero();
  });
  EXPECT_EQ(folded2, Poly(Coeffs(std::move(terms))));

  GF7 r3 = GF7::Random();
  Poly folded3 = folded2.Fold<false>(r3);
  terms = {{0, folded2[0] + r3 * folded2[1]}};
  base::EraseIf(terms, [](const UnivariateTerm<GF7>& term) {
    return term.coefficient.IsZero();
  });
  EXPECT_EQ(folded3, Poly(Coeffs(std::move(terms))));
}

TEST_F(UnivariateSparsePolynomialTest, Copyable) {
  Poly expected = Poly::Random(kMaxDegree);

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  Poly value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST_F(UnivariateSparsePolynomialTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(Poly(), Poly::Zero(), Poly::One(),
                      Poly::Random(kMaxDegree), Poly::Random(kMaxDegree))));
}

TEST_F(UnivariateSparsePolynomialTest, JsonValueConverter) {
  Poly expected_poly(Coeffs({{0, GF7(1)}, {1, GF7(2)}, {4, GF7(3)}}));
  std::string expected_json =
      R"({"coefficients":{"terms":[{"degree":0,"coefficient":{"value":"0x1"}},{"degree":1,"coefficient":{"value":"0x2"}},{"degree":4,"coefficient":{"value":"0x3"}}]}})";

  Poly poly;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &poly, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(poly, expected_poly);

  std::string json = base::WriteToJson(poly);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
