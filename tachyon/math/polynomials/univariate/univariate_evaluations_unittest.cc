#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
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
  EXPECT_TRUE(Poly::One().IsOne());
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
  Poly r = Poly::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != Poly::Random()) {
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
        EXPECT_EQ(*test.poly[i], GF7(test.evaluations[i]));
      } else {
        EXPECT_EQ(test.poly[i], nullptr);
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

TEST_F(UnivariateEvaluationsTest, Copyable) {
  base::VectorBuffer buf;
  buf.Write(polys_[0]);

  buf.set_buffer_offset(0);
  Poly value;
  buf.Read(&value);

  EXPECT_EQ(polys_[0], value);
}

}  // namespace tachyon::math
