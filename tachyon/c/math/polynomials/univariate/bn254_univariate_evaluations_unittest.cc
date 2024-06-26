#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluations_type_traits.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::math {

namespace {

constexpr size_t kDegree = 5;

class UnivariateEvaluationsTest : public FiniteFieldTest<bn254::Fr> {
 public:
  using Evals = UnivariateEvaluations<bn254::Fr, c::math::kMaxDegree>;

  void SetUp() override {
    Evals* cpp_evals = new Evals(Evals::Random(kDegree));
    evals_ = c::base::c_cast(cpp_evals);
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
  // NOTE(chokobole): It's safe to access since we created |kDegree| |evals_|.
  c::base::native_cast(*evals_).at(0) += bn254::Fr::One();
  EXPECT_NE((c::base::native_cast(*evals_))[0],
            (c::base::native_cast(*evals_clone))[0]);
  tachyon_bn254_univariate_evaluations_destroy(evals_clone);
}

TEST_F(UnivariateEvaluationsTest, Len) {
  EXPECT_EQ(tachyon_bn254_univariate_evaluations_len(evals_), kDegree + 1);
}

TEST_F(UnivariateEvaluationsTest, SetValue) {
  bn254::Fr cpp_value = bn254::Fr::Random();
  const tachyon_bn254_fr& value = c::base::c_cast(cpp_value);
  tachyon_bn254_univariate_evaluations_set_value(evals_, 0, &value);
  // NOTE(chokobole): It's safe to access since we created |kDegree| |evals_|.
  EXPECT_EQ(c::base::native_cast(*evals_)[0], cpp_value);
}

}  // namespace tachyon::math
