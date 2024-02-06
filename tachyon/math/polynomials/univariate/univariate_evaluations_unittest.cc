#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

const size_t kMaxDegree = 4;

using Poly = UnivariateEvaluations<GF7, kMaxDegree>;

class UnivariateEvaluationsTest : public testing::Test {
 public:
  static void SetUpTestSuite() { GF7::Init(); }

  void SetUp() override {
    polys_.push_back(Poly({GF7(3), GF7(6), GF7(4), GF7(6), GF7(6)}));
    polys_.push_back(Poly({GF7(3), GF7(4), GF7(5), GF7(0), GF7(2)}));
    polys_.push_back(Poly({GF7(2), GF7(5), GF7(5), GF7(2), GF7(1)}));
    polys_.push_back(Poly({GF7(1), GF7(5), GF7(3), GF7(6), GF7(6)}));
    polys_.push_back(Poly({GF7(0), GF7(0), GF7(0), GF7(0), GF7(0)}));
    polys_.push_back(Poly::Zero());
    polys_.push_back(Poly({GF7(1), GF7(1), GF7(1), GF7(1), GF7(1)}));
  }

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(UnivariateEvaluationsTest, IsZero) {
  EXPECT_TRUE(Poly().IsZero());
  EXPECT_TRUE(Poly::Zero().IsZero());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    if (i == polys_.size() - 2 || i == polys_.size() - 3) {
      EXPECT_TRUE(polys_[i].IsZero());
    } else {
      EXPECT_FALSE(polys_[i].IsZero());
    }
  }
}

TEST_F(UnivariateEvaluationsTest, IsOne) {
  EXPECT_FALSE(Poly().IsOne());
  EXPECT_TRUE(Poly::One(kMaxDegree).IsOne());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    if (i == polys_.size() - 1) {
      EXPECT_TRUE(polys_[i].IsOne());
    } else {
      EXPECT_FALSE(polys_[i].IsOne());
    }
  }
}

TEST_F(UnivariateEvaluationsTest, Random) {
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

TEST_F(UnivariateEvaluationsTest, IndexingOperator) {
  struct {
    const Poly& poly;
    std::vector<int> evaluations;
  } tests[] = {{polys_[0], {3, 6, 4, 6, 6}}, {polys_[1], {3, 4, 5, 0, 2}},
               {polys_[2], {2, 5, 5, 2, 1}}, {polys_[3], {1, 5, 3, 6, 6}},
               {polys_[4], {0, 0, 0, 0, 0}}, {polys_[5], {}},
               {polys_[6], {1, 1, 1, 1, 1}}};

  for (const auto& test : tests) {
    for (size_t i = 0; i < kMaxDegree; ++i) {
      if (i < test.evaluations.size()) {
        EXPECT_EQ(test.poly[i], GF7(test.evaluations[i]));
      } else {
        EXPECT_EQ(test.poly[i], GF7::Zero());
      }
    }
  }
}

TEST_F(UnivariateEvaluationsTest, EqualityOperators) {
  struct {
    const Poly& a;
    const Poly& b;
    bool equal;
  } tests[] = {
      {polys_[0], polys_[1], false},
      {polys_[0], polys_[4], false},
      {polys_[4], polys_[5], true},
  };

  for (const auto& test : tests) {
    if (test.equal) {
      EXPECT_EQ(test.a, test.b);
      EXPECT_EQ(test.b, test.a);
    } else {
      EXPECT_NE(test.a, test.b);
      EXPECT_NE(test.b, test.a);
    }
  }
}

TEST_F(UnivariateEvaluationsTest, AdditiveOperators) {
  struct {
    const Poly& a;
    const Poly& b;
    Poly sum;
    Poly amb;
    Poly bma;
  } tests[] = {
      {
          polys_[0],
          polys_[3],
          Poly({GF7(4), GF7(4), GF7(0), GF7(5), GF7(5)}),
          Poly({GF7(2), GF7(1), GF7(1), GF7(0), GF7(0)}),
          Poly({GF7(5), GF7(6), GF7(6), GF7(0), GF7(0)}),
      },
      {
          polys_[0],
          polys_[4],
          polys_[0],
          polys_[0],
          -polys_[0],
      },
      {
          polys_[0],
          polys_[5],
          polys_[0],
          polys_[0],
          -polys_[0],
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    Poly tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(UnivariateEvaluationsTest, MultiplicativeOperators) {
  struct {
    const Poly& a;
    const Poly& b;
    Poly mul;
    Poly adb;
    Poly bda;
  } tests[] = {
      {
          polys_[0],
          polys_[3],
          Poly({GF7(3), GF7(2), GF7(5), GF7(1), GF7(1)}),
          Poly({GF7(3), GF7(4), GF7(6), GF7(1), GF7(1)}),
          Poly({GF7(5), GF7(2), GF7(6), GF7(1), GF7(1)}),
      },
      {
          polys_[0],
          polys_[4],
          Poly::Zero(),
          // NOTE(chokobole): division by zero.
          Poly::Zero(),
          Poly::Zero(),
      },
      {
          polys_[0],
          polys_[5],
          Poly::Zero(),
          // NOTE(chokobole): division by zero.
          Poly::Zero(),
          Poly::Zero(),
      },
      {
          polys_[0],
          polys_[6],
          polys_[0],
          polys_[0],
          Poly({GF7(5), GF7(6), GF7(2), GF7(6), GF7(6)}),
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    if (!test.b.IsZero()) {
      EXPECT_EQ(test.a / test.b, test.adb);
    }
    if (!test.a.IsZero()) {
      EXPECT_EQ(test.b / test.a, test.bda);
    }
    Poly tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    if (!test.b.IsZero()) {
      tmp = test.a;
      tmp /= test.b;
      EXPECT_EQ(tmp, test.adb);
    }
  }
}

TEST_F(UnivariateEvaluationsTest, MulScalar) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 scalar = GF7::Random();

  std::vector<GF7> expected_evals;
  const std::vector<GF7>& evals = poly.evaluations();
  expected_evals.reserve(evals.size());
  for (size_t i = 0; i < evals.size(); ++i) {
    expected_evals.push_back(evals[i] * scalar);
  }

  Poly actual = poly * scalar;
  Poly expected(std::move(expected_evals));
  EXPECT_EQ(actual, expected);
  poly *= scalar;
  EXPECT_EQ(poly, expected);
}

TEST_F(UnivariateEvaluationsTest, DivScalar) {
  Poly poly = Poly::Random(kMaxDegree);
  GF7 scalar = GF7::Random();
  while (scalar.IsZero()) {
    scalar = GF7::Random();
  }

  std::vector<GF7> expected_evals;
  const std::vector<GF7>& evals = poly.evaluations();
  expected_evals.reserve(evals.size());
  for (size_t i = 0; i < evals.size(); ++i) {
    expected_evals.push_back(evals[i] / scalar);
  }

  Poly actual = poly / scalar;
  Poly expected(std::move(expected_evals));
  EXPECT_EQ(actual, expected);
  poly /= scalar;
  EXPECT_EQ(poly, expected);
}

TEST_F(UnivariateEvaluationsTest, Copyable) {
  Poly expected = Poly::Random(kMaxDegree);

  std::vector<uint8_t> vec;
  vec.resize(base::EstimateSize(expected));
  base::Buffer write_buf(vec.data(), vec.size());
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  Poly value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST_F(UnivariateEvaluationsTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(Poly(), Poly::Zero(),
                      Poly({base::CreateVector(kMaxDegree + 1, GF7::Zero())}),
                      Poly::One(kMaxDegree), Poly::Random(kMaxDegree),
                      Poly::Random(kMaxDegree))));
}

TEST_F(UnivariateEvaluationsTest, JsonValueConverter) {
  Poly expected_poly({GF7(1), GF7(2), GF7(3), GF7(4), GF7(5)});
  std::string expected_json =
      R"({"evaluations":[{"value":"0x1"},{"value":"0x2"},{"value":"0x3"},{"value":"0x4"},{"value":"0x5"}]})";

  Poly poly;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &poly, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(poly, expected_poly);

  std::string json = base::WriteToJson(poly);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
