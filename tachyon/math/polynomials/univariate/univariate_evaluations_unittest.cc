#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

const size_t kMaxSize = 6;

using Poly = UnivariateEvaluations<GF7, kMaxSize>;

class UnivariateEvaluationsTest : public testing::Test {
 public:
  static void SetUpTestSuite() { GF7::Init(); }

  void SetUp() override {
    polys_.push_back(Poly({GF7(3), GF7(6), GF7(4), GF7(6), GF7(6)}));
    polys_.push_back(Poly({GF7(3)}));
    polys_.push_back(Poly({GF7(2), GF7(5), GF7(5), GF7(2)}));
    polys_.push_back(Poly({GF7(1), GF7(5), GF7(3), GF7(6), GF7(6)}));
    polys_.push_back(Poly::Zero(1));
  }

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(UnivariateEvaluationsTest, IsZero) {
  EXPECT_TRUE(Poly().IsZero());
  EXPECT_TRUE(Poly::Zero(kMaxSize).IsZero());
  EXPECT_TRUE(Poly({GF7(0)}).IsZero());
  EXPECT_FALSE(Poly({GF7(0), GF7(1)}).IsZero());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    EXPECT_FALSE(polys_[i].IsZero());
  }
  EXPECT_TRUE(polys_[polys_.size() - 1].IsZero());
}

TEST_F(UnivariateEvaluationsTest, IsOne) {
  EXPECT_TRUE(Poly::One(kMaxSize).IsOne());
  EXPECT_TRUE(Poly({GF7(1)}).IsOne());
  EXPECT_FALSE(Poly({GF7(0), GF7(1)}).IsZero());
  for (size_t i = 0; i < polys_.size(); ++i) {
    EXPECT_FALSE(polys_[i].IsOne());
  }
}

TEST_F(UnivariateEvaluationsTest, Random) {
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

TEST_F(UnivariateEvaluationsTest, IndexingOperator) {
  struct {
    const Poly& poly;
    std::vector<int> evaluations;
  } tests[] = {
      {polys_[0], {3, 6, 4, 6, 6}}, {polys_[1], {3}}, {polys_[2], {2, 5, 5, 2}},
      {polys_[3], {1, 5, 3, 6, 6}}, {polys_[4], {0}},
  };

  for (const auto& test : tests) {
    for (size_t i = 0; i < kMaxSize; ++i) {
      if (i < test.evaluations.size()) {
        EXPECT_EQ(*test.poly[i], GF7(test.evaluations[i]));
      } else {
        EXPECT_EQ(test.poly[i], nullptr);
      }
    }
  }
}

TEST_F(UnivariateEvaluationsTest, Degree) {
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
          polys_[1],
          polys_[4],
          Poly({GF7(3)}),
          Poly({GF7(3)}),
          Poly({GF7(4)}),
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
          polys_[1],
          polys_[4],
          Poly({GF7(0)}),
          Poly({GF7(0)}),
          Poly({GF7(0)}),
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

TEST_F(UnivariateEvaluationsTest, Copyable) {
  Poly expected({GF7(4), GF7(4), GF7(0), GF7(5), GF7(5)});
  Poly value;

  base::VectorBuffer buf;
  buf.Write(expected);

  buf.set_buffer_offset(0);
  buf.Read(&value);

  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::math
