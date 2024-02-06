#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_prime_field_traits.h"
#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::math {

namespace {

constexpr size_t kDegree = 5;

class UnivariateEvaluationsTest : public testing::Test {
 public:
  using Evals = UnivariateEvaluations<bn254::Fr, c::math::kMaxDegree>;

  static void SetUpTestSuite() { bn254::Fr::Init(); }

  void SetUp() override {
    Evals* cpp_evals = new Evals(Evals::Random(kDegree));
    evals_ = reinterpret_cast<tachyon_bn254_univariate_evaluations*>(cpp_evals);
  }

  void TearDown() override {
    tachyon_bn254_univariate_evaluations_destroy(evals_);
  }

 protected:
  tachyon_bn254_univariate_evaluations* evals_;
};

}  // namespace

TEST_F(UnivariateEvaluationsTest, Clone) {
  tachyon_bn254_univariate_evaluations* evals_clone =
      tachyon_bn254_univariate_evaluations_clone(evals_);
  *reinterpret_cast<Evals&>(*evals_)[0] += bn254::Fr::One();
  EXPECT_NE((reinterpret_cast<Evals&>(*evals_))[0],
            (reinterpret_cast<Evals&>(*evals_clone))[0]);
  tachyon_bn254_univariate_evaluations_destroy(evals_clone);
}

TEST_F(UnivariateEvaluationsTest, Len) {
  EXPECT_EQ(tachyon_bn254_univariate_evaluations_len(evals_), kDegree + 1);
}

TEST_F(UnivariateEvaluationsTest, SetValue) {
  bn254::Fr cpp_value = bn254::Fr::Random();
  tachyon_bn254_fr value = cc::math::ToCPrimeField(cpp_value);
  tachyon_bn254_univariate_evaluations_set_value(evals_, 0, &value);
  EXPECT_EQ(*reinterpret_cast<Evals&>(*evals_)[0], cpp_value);
}

}  // namespace tachyon::math
