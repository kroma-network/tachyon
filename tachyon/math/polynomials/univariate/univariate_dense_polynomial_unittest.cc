#include <tuple>

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

namespace {

const size_t kMaxDegree = 5;

using Poly = UnivariateDensePolynomial<GF7, kMaxDegree>;
using Coeffs = UnivariateDenseCoefficients<GF7, kMaxDegree>;

class UnivariateDensePolynomialTest : public FiniteFieldTest<GF7> {
 public:
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
  EXPECT_TRUE(Poly(Coeffs({GF7(0), GF7(0)}, true)).IsZero());
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
        EXPECT_EQ(test.poly[i], GF7(test.coefficients[i]));
      } else {
        EXPECT_EQ(test.poly[i], GF7::Zero());
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
          a,
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
    } else {
      ASSERT_FALSE(test.a / test.b);
    }
    if (!test.a.IsZero()) {
      EXPECT_EQ(test.b / test.a, test.bda);
      EXPECT_EQ(test.b % test.a, test.bma);
    } else {
      ASSERT_FALSE(test.b / test.a);
    }
    EXPECT_EQ(test.a * b_sparse, test.mul);
    EXPECT_EQ(test.b * a_sparse, test.mul);
    if (!b_sparse.IsZero()) {
      EXPECT_EQ(test.a / b_sparse, test.adb);
      EXPECT_EQ(test.a % b_sparse, test.amb);
    } else {
      ASSERT_FALSE(test.a / b_sparse);
    }
    if (!a_sparse.IsZero()) {
      EXPECT_EQ(test.b / a_sparse, test.bda);
      EXPECT_EQ(test.b % a_sparse, test.bma);
    } else {
      ASSERT_FALSE(test.b / a_sparse);
    }

    Poly tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    if (!test.b.IsZero()) {
      ASSERT_TRUE(tmp /= test.b);
      EXPECT_EQ(tmp, test.a);
    } else {
      ASSERT_FALSE(tmp /= test.b);
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
  Poly expected(Coeffs(std::move(expected_coeffs), true));
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
    expected_coeffs.push_back(unwrap(coeffs[i] / scalar));
  }

  Poly actual = unwrap(poly / scalar);
  Poly expected(Coeffs(std::move(expected_coeffs)));
  EXPECT_EQ(actual, expected);
  ASSERT_TRUE(poly /= scalar);
  EXPECT_EQ(poly, expected);
}

TEST_F(UnivariateDensePolynomialTest, FromRoots) {
  // poly = x⁴ + 2x² + 4 = (x - 1)(x - 2)(x + 1)(x + 2)
  Poly poly = Poly(Coeffs({GF7(4), GF7::Zero(), GF7(2), GF7::Zero(), GF7(1)}));
  std::vector<GF7> roots = {GF7(1), GF7(2), GF7(6), GF7(5)};
  EXPECT_EQ(Poly::FromRoots(roots), poly);
}

TEST_F(UnivariateDensePolynomialTest, EvaluateVanishingPolyByRoots) {
  // poly = x⁴ + 2x² + 4 = (x - 1)(x - 2)(x + 1)(x + 2)
  Poly poly = Poly(Coeffs({GF7(4), GF7::Zero(), GF7(2), GF7::Zero(), GF7(1)}));
  std::vector<GF7> roots = {GF7(1), GF7(2), GF7(6), GF7(5)};
  GF7 point = GF7::Random();
  EXPECT_EQ(Poly::EvaluateVanishingPolyByRoots(roots, point),
            poly.Evaluate(point));
}

TEST_F(UnivariateDensePolynomialTest, Fold) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 r = GF7::Random();
  Poly folded = poly.Fold(r);
  EXPECT_EQ(folded, Poly(Coeffs({poly[0] + r * poly[1], poly[2] + r * poly[3],
                                 poly[4] + r * poly[5]},
                                true)));

  GF7 r2 = GF7::Random();
  Poly folded2 = folded.Fold(r2);
  EXPECT_EQ(folded2,
            Poly(Coeffs({folded[0] + r2 * folded[1], folded[2]}, true)));

  GF7 r3 = GF7::Random();
  Poly folded3 = folded2.Fold(r3);
  EXPECT_EQ(folded3, Poly(Coeffs({folded2[0] + r3 * folded2[1]}, true)));
}

TEST_F(UnivariateDensePolynomialTest, Copyable) {
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

TEST_F(UnivariateDensePolynomialTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(Poly(), Poly::Zero(), Poly::One(),
                      Poly::Random(kMaxDegree), Poly::Random(kMaxDegree))));
}

TEST_F(UnivariateDensePolynomialTest, JsonValueConverter) {
  Poly expected_poly(Coeffs({GF7(1), GF7(2), GF7(3), GF7(4), GF7(5)}));
  std::string expected_json =
      R"({"coefficients":{"coefficients":[1,2,3,4,5]}})";

  Poly poly;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &poly, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(poly, expected_poly);

  std::string json = base::WriteToJson(poly);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
