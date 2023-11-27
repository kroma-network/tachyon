#include "tachyon/math/polynomials/multivariate/multilinear_sparse_evaluations.h"

#include <optional>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/multivariate/multilinear_extension.h"

namespace tachyon::math {

namespace {
const size_t kMaxDegree = 3;
const size_t kNumVars = 3;

using Poly = MultilinearExtension<
    MultilinearSparseEvaluations<GF7, kMaxDegree, kNumVars>>;
using Evals = MultilinearSparseEvaluations<GF7, kMaxDegree, kNumVars>;

class MultilinearSparseEvaluationsTest : public testing::Test {
 public:
  static void SetUpTestSuite() { GF7::Init(); }

  MultilinearSparseEvaluationsTest() {
    polys_.push_back(Poly(Evals({{1, GF7(2)}, {0, GF7(3)}})));
    polys_.push_back(Poly(Evals({{2, GF7(3)}, {1, GF7(2)}})));
    polys_.push_back(
        Poly(Evals({{0, GF7(3)}, {2, GF7(2)}, {1, GF7(3)}, {5, GF7(2)}})));
    polys_.push_back(Poly(Evals({{2, GF7(3)}, {1, GF7(2)}})));
  }
  MultilinearSparseEvaluationsTest(const MultilinearSparseEvaluationsTest&) =
      delete;
  MultilinearSparseEvaluationsTest& operator=(
      const MultilinearSparseEvaluationsTest&) = delete;
  ~MultilinearSparseEvaluationsTest() override = default;

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(MultilinearSparseEvaluationsTest, IsZero) {
  EXPECT_TRUE(Poly().IsZero());
  EXPECT_TRUE(Poly::Zero(3).IsZero());
  EXPECT_TRUE(Poly(Evals({{0, GF7(0)}})).IsZero());
  for (size_t i = 0; i < polys_.size(); ++i) {
    EXPECT_FALSE(Poly(polys_[i]).IsZero());
  }
}

TEST_F(MultilinearSparseEvaluationsTest, IsOne) {
  EXPECT_TRUE(Poly::One(kMaxDegree).IsOne());
  EXPECT_TRUE(Poly(Evals({{1, GF7(1)}})).IsOne());
  for (size_t i = 0; i < polys_.size(); ++i) {
    EXPECT_FALSE(polys_[i].IsOne());
  }
}

TEST_F(MultilinearSparseEvaluationsTest, Random) {
  bool success = false;
  absl::BitGen rnd;
  Poly r = Poly(Evals::Random(kNumVars, rnd));
  for (size_t i = 0; i < 100; ++i) {
    if (r != Poly(Evals::Random(kNumVars, rnd))) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(MultilinearSparseEvaluationsTest, IndexingOperator) {
  struct {
    const Poly& poly;
    std::vector<std::optional<int>> evaluations;

  } tests[] = {
      {polys_[0], {3, 2}}, {polys_[1], {std::nullopt, 2, 3}},
      // {polys_[0], {3, 2, 2}},
  };

  for (const auto& test : tests) {
    for (size_t i = 0; i < kMaxDegree; ++i) {
      if (i < test.evaluations.size()) {
        if (test.evaluations[i].has_value()) {
          EXPECT_EQ(*test.poly[i], GF7(test.evaluations[i].value()));
        } else {
          EXPECT_EQ(test.poly[i], nullptr);
        }
      } else {
        EXPECT_EQ(test.poly[i], nullptr);
      }
    }
  }
}

TEST_F(MultilinearSparseEvaluationsTest, Degree) {
  absl::BitGen rnd;
  struct {
    const Poly& poly;
    size_t degree;
  } tests[] = {{polys_[0], 0}, {polys_[1], 1}, {polys_[2], 3}};

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Degree(), test.degree);
  }

  EXPECT_LE(Poly(Evals::Random(kNumVars, rnd)).Degree(), kMaxDegree);
}

TEST_F(MultilinearSparseEvaluationsTest, Evaluate) {
  auto convert_to_le = [](size_t number, size_t degree) {
    std::vector<GF7> ret;
    for (size_t i = 0; i < degree; ++i) {
      ret.push_back(GF7(uint64_t{(number & (size_t{1} << i)) != 0}));
    }
    return ret;
  };

  for (const Poly& poly : polys_) {
    size_t degree = poly.Degree();
    for (size_t i = 0; i < (size_t{1} << degree); ++i) {
      const GF7* elem = poly[i];
      if (!elem) break;
      std::vector<GF7> point = convert_to_le(i, degree);
      EXPECT_EQ(poly.Evaluate(point), *elem);
    }
  }
}

TEST_F(MultilinearSparseEvaluationsTest, ToString) {
  struct {
    const Poly& poly;
    std::string_view expected;
  } tests[] = {
      {polys_[0], "[(0, 3), (1, 2)]"},
      {polys_[1], "[(1, 2), (2, 3)]"},
      {polys_[2], "[(0, 3), (1, 3), (2, 2), (5, 2)]"},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.ToString(), test.expected);
  }
}

TEST_F(MultilinearSparseEvaluationsTest, ToDense) {
  struct {
    const Poly& poly;
    std::string_view expected;
  } tests[] = {
      {polys_[0], "[3, 2, 0, 0, 0, 0, 0, 0]"},
      {polys_[1], "[0, 2, 3, 0, 0, 0, 0, 0]"},
      {polys_[2], "[3, 3, 2, 0, 0, 2, 0, 0]"},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.ToDense().ToString(), test.expected);
  }
}

TEST_F(MultilinearSparseEvaluationsTest, AdditiveOperators) {
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
          Poly(Evals({{0, GF7(3)}, {1, GF7(4)}, {2, GF7(3)}})),
          Poly(Evals({{0, GF7(3)}, {2, GF7(4)}})),
          Poly(Evals({{0, GF7(4)}, {2, GF7(3)}})),
      },

      {
          Poly::Zero(3),
          Poly(Evals({{0, GF7(3)}, {1, GF7(4)}, {2, GF7(3)}})),
          Poly(Evals({{0, GF7(3)}, {1, GF7(4)}, {2, GF7(3)}})),
          Poly(Evals({{0, GF7(4)}, {1, GF7(3)}, {2, GF7(4)}})),
          Poly(Evals({{0, GF7(3)}, {1, GF7(4)}, {2, GF7(3)}})),
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

TEST_F(MultilinearSparseEvaluationsTest, MultiplicativeOperators) {
  struct {
    const Poly& a;
    const Poly& b;
    Poly mul;
    Poly adb;
    Poly bda;
  } tests[] = {
      {
          polys_[1],
          polys_[3],
          Poly(Evals({{1, GF7(4)}, {2, GF7(2)}})),
          Poly(Evals({{1, GF7(1)}, {2, GF7(1)}})),
          Poly(Evals({{1, GF7(1)}, {2, GF7(1)}})),
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

TEST_F(MultilinearSparseEvaluationsTest, DivisionByZero) {
  struct {
    const Poly& a;
    const Poly& b;

  } tests[] = {
      {
          polys_[0],
          polys_[1],
      },
  };

  for (const auto& test : tests) {
    EXPECT_THROW({ test.a / test.b; }, std::runtime_error);
    EXPECT_THROW({ test.b / test.a; }, std::runtime_error);

    Poly tmp = test.a;
    EXPECT_THROW({ tmp /= test.b; }, std::runtime_error);
  }
}

}  // namespace tachyon::math
