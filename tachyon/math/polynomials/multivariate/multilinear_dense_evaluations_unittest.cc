#include "tachyon/math/polynomials/multivariate/multilinear_dense_evaluations.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/multivariate/multilinear_extension.h"

namespace tachyon::math {

namespace {

const size_t kMaxDegree = 4;

using Point = std::vector<GF7>;
using Poly = MultilinearExtension<MultilinearDenseEvaluations<GF7, kMaxDegree>>;
using Evals = MultilinearDenseEvaluations<GF7, kMaxDegree>;

class MultilinearDenseEvaluationsTest : public FiniteFieldTest<GF7> {
 public:
  void SetUp() override {
    polys_.push_back(Poly(Evals({GF7(2), GF7(3)})));
    polys_.push_back(Poly(Evals({GF7(4), GF7(2)})));
    polys_.push_back(Poly(Evals(
        {GF7(2), GF7(3), GF7(2), GF7(6), GF7(5), GF7(1), GF7(4), GF7(2)})));
    polys_.push_back(Poly(Evals(
        {GF7(3), GF7(1), GF7(1), GF7(4), GF7(2), GF7(3), GF7(2), GF7(5)})));
  }

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(MultilinearDenseEvaluationsTest, IsZero) {
  EXPECT_TRUE(Poly().IsZero());
  EXPECT_TRUE(Poly::Zero().IsZero());
  EXPECT_TRUE(Poly(Evals({GF7(0)})).IsZero());
  for (size_t i = 0; i < polys_.size(); ++i) {
    EXPECT_FALSE(Poly(polys_[i]).IsZero());
  }
}

TEST_F(MultilinearDenseEvaluationsTest, IsOne) {
  EXPECT_TRUE(Poly::One(0).IsOne());
  EXPECT_TRUE(Poly(Evals({GF7(1)})).IsOne());
  for (size_t i = 0; i < polys_.size(); ++i) {
    EXPECT_FALSE(polys_[i].IsOne());
  }
}

TEST_F(MultilinearDenseEvaluationsTest, Random) {
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

TEST_F(MultilinearDenseEvaluationsTest, IndexingOperator) {
  struct {
    const Poly& poly;
    std::vector<int> evaluations;
  } tests[] = {
      {polys_[0], {2, 3}},
      {polys_[1], {4, 2}},
      {polys_[2], {2, 3, 2, 6, 5, 1, 4, 2}},
      {polys_[3], {3, 1, 1, 4, 2, 3, 2, 5}},
  };

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

TEST_F(MultilinearDenseEvaluationsTest, Degree) {
  struct {
    const Poly& poly;
    size_t degree;
  } tests[] = {{polys_[0], 1}, {polys_[1], 1}, {polys_[2], 3}, {polys_[3], 3}};

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Degree(), test.degree);
  }
  EXPECT_LE(Poly::Random(kMaxDegree).Degree(), kMaxDegree);
}

TEST_F(MultilinearDenseEvaluationsTest, Evaluate) {
  std::function<Point(size_t, size_t)> convert_to_le = [](size_t number,
                                                          size_t degree) {
    Point ret;
    for (size_t i = 0; i < degree; ++i) {
      ret.push_back(GF7(uint64_t{(number & (size_t{1} << i)) != 0}));
    }
    return ret;
  };

  for (const Poly& poly : polys_) {
    size_t degree = poly.Degree();
    for (size_t i = 0; i < (size_t{1} << degree); ++i) {
      const GF7& elem = poly[i];
      if (elem.IsZero()) break;
      Point point = convert_to_le(i, degree);
      EXPECT_EQ(poly.Evaluate(point), elem);
    }
  }
}

TEST_F(MultilinearDenseEvaluationsTest, ToString) {
  struct {
    const Poly& poly;
    std::string_view expected;
  } tests[] = {
      {polys_[0], "[2, 3]"},
      {polys_[1], "[4, 2]"},
      {polys_[2], "[2, 3, 2, 6, 5, 1, 4, 2]"},
      {polys_[3], "[3, 1, 1, 4, 2, 3, 2, 5]"},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.ToString(), test.expected);
  }
}

TEST_F(MultilinearDenseEvaluationsTest, AdditiveOperators) {
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
          Poly(Evals({GF7(6), GF7(5)})),
          Poly(Evals({GF7(5), GF7(1)})),
          Poly(Evals({GF7(2), GF7(6)})),
      },
      {
          polys_[2],
          polys_[3],
          Poly(Evals({GF7(5), GF7(4), GF7(3), GF7(3), GF7(0), GF7(4), GF7(6),
                      GF7(0)})),
          Poly(Evals({GF7(6), GF7(2), GF7(1), GF7(2), GF7(3), GF7(5), GF7(2),
                      GF7(4)})),
          Poly(Evals({GF7(1), GF7(5), GF7(6), GF7(5), GF7(4), GF7(2), GF7(5),
                      GF7(3)})),
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

TEST_F(MultilinearDenseEvaluationsTest, MultiplicativeOperators) {
  struct {
    const Poly& a;
    const Poly& b;
    Poly mul;
    Poly adb;
    Poly bda;
  } tests[] = {
      {
          polys_[0],
          polys_[1],
          Poly(Evals({GF7(1), GF7(6)})),
          Poly(Evals({GF7(4), GF7(5)})),
          Poly(Evals({GF7(2), GF7(3)})),
      },
      {
          polys_[2],
          polys_[3],
          Poly(Evals({GF7(6), GF7(3), GF7(2), GF7(3), GF7(3), GF7(3), GF7(1),
                      GF7(3)})),
          Poly(Evals({GF7(3), GF7(3), GF7(2), GF7(5), GF7(6), GF7(5), GF7(2),
                      GF7(6)})),
          Poly(Evals({GF7(5), GF7(5), GF7(4), GF7(3), GF7(6), GF7(3), GF7(4),
                      GF7(6)})),
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    Poly tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    tmp /= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(MultilinearDenseEvaluationsTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(Poly(), Poly::Zero(),
                      Poly(Evals(std::vector<GF7>(size_t{1} << kMaxDegree))),
                      Poly::One(kMaxDegree), Poly::Random(kMaxDegree),
                      Poly::Random(kMaxDegree))));
}

}  // namespace tachyon::math
