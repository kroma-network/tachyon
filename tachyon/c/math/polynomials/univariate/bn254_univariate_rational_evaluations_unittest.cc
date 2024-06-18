#include "tachyon/c/math/polynomials/univariate/bn254_univariate_rational_evaluations.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::math {

namespace {

constexpr size_t kDegree = 5;

class UnivariateRationalEvaluationsTest : public FiniteFieldTest<bn254::Fr> {
 public:
  using Evals = UnivariateEvaluations<bn254::Fr, c::math::kMaxDegree>;
  using RationalEvals =
      UnivariateEvaluations<RationalField<bn254::Fr>, c::math::kMaxDegree>;

  void SetUp() override {
    RationalEvals* cpp_evals =
        new RationalEvals(RationalEvals::Random(kDegree));
    evals_ = reinterpret_cast<tachyon_bn254_univariate_rational_evaluations*>(
        cpp_evals);
  }

  void TearDown() override {
    tachyon_bn254_univariate_rational_evaluations_destroy(evals_);
  }

 protected:
  tachyon_bn254_univariate_rational_evaluations* evals_;
};

}  // namespace

TEST_F(UnivariateRationalEvaluationsTest, Clone) {
  tachyon_bn254_univariate_rational_evaluations* evals_clone =
      tachyon_bn254_univariate_rational_evaluations_clone(evals_);
  // NOTE(chokobole): It's safe to access since we created |kDegree| |evals_|.
  reinterpret_cast<RationalEvals&>(*evals_).at(0) +=
      RationalField<bn254::Fr>::One();
  EXPECT_NE((reinterpret_cast<RationalEvals&>(*evals_))[0],
            (reinterpret_cast<RationalEvals&>(*evals_clone))[0]);
  tachyon_bn254_univariate_rational_evaluations_destroy(evals_clone);
}

TEST_F(UnivariateRationalEvaluationsTest, Len) {
  EXPECT_EQ(tachyon_bn254_univariate_rational_evaluations_len(evals_),
            kDegree + 1);
}

TEST_F(UnivariateRationalEvaluationsTest, SetZero) {
  tachyon_bn254_univariate_rational_evaluations_set_zero(evals_, 0);
  EXPECT_TRUE(reinterpret_cast<RationalEvals&>(*evals_)[0].IsZero());
}

TEST_F(UnivariateRationalEvaluationsTest, SetTrivial) {
  RationalField<bn254::Fr> expected =
      RationalField<bn254::Fr>(bn254::Fr::Random());
  const tachyon_bn254_fr& numerator = c::base::c_cast(expected.numerator());
  tachyon_bn254_univariate_rational_evaluations_set_trivial(evals_, 0,
                                                            &numerator);
  EXPECT_EQ(reinterpret_cast<RationalEvals&>(*evals_)[0], expected);
}

TEST_F(UnivariateRationalEvaluationsTest, SetRational) {
  RationalField<bn254::Fr> expected = RationalField<bn254::Fr>::Random();
  const tachyon_bn254_fr& numerator = c::base::c_cast(expected.numerator());
  const tachyon_bn254_fr& denominator = c::base::c_cast(expected.denominator());
  tachyon_bn254_univariate_rational_evaluations_set_rational(
      evals_, 0, &numerator, &denominator);
  EXPECT_EQ(reinterpret_cast<RationalEvals&>(*evals_)[0], expected);
}

TEST_F(UnivariateRationalEvaluationsTest, BatchEvaluate) {
  std::vector<RationalField<bn254::Fr>> rational_values = base::CreateVector(
      kDegree + 1, []() { return RationalField<bn254::Fr>::Random(); });
  for (size_t i = 0; i < rational_values.size(); ++i) {
    const tachyon_bn254_fr& numerator =
        c::base::c_cast(rational_values[i].numerator());
    const tachyon_bn254_fr& denominator =
        c::base::c_cast(rational_values[i].denominator());
    tachyon_bn254_univariate_rational_evaluations_set_rational(
        evals_, i, &numerator, &denominator);
  }
  tachyon_bn254_univariate_evaluations* evaluated =
      tachyon_bn254_univariate_rational_evaluations_batch_evaluate(evals_);
  std::vector<bn254::Fr> values;
  values.resize(rational_values.size());
  CHECK(RationalField<bn254::Fr>::BatchEvaluate(rational_values, &values));
  EXPECT_EQ(reinterpret_cast<Evals&>(*evaluated).evaluations(), values);
}

}  // namespace tachyon::math
