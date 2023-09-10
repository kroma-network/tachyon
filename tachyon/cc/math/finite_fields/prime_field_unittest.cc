#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_prime_field_traits.h"
#include "tachyon/cc/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::cc::math {

namespace {

class PrimeFieldTest : public testing::Test {
 public:
  void SetUp() override {
    a_ = tachyon::math::bn254::Fr::Random();
    b_ = tachyon::math::bn254::Fr::Random();

    cc_a_ = bn254::Fr(ToCPrimeField(a_));
    cc_b_ = bn254::Fr(ToCPrimeField(b_));
  }

 protected:
  tachyon::math::bn254::Fr a_;
  tachyon::math::bn254::Fr b_;
  bn254::Fr cc_a_;
  bn254::Fr cc_b_;
};

}  // namespace

TEST_F(PrimeFieldTest, Zero) {
  bn254::Fr zero = bn254::Fr::Zero();
  EXPECT_TRUE(cc::math::ToPrimeField(zero.value()).IsZero());
}

TEST_F(PrimeFieldTest, One) {
  bn254::Fr one = bn254::Fr::One();
  EXPECT_TRUE(cc::math::ToPrimeField(one.value()).IsOne());
}

TEST_F(PrimeFieldTest, Random) {
  bn254::Fr random = bn254::Fr::Random();
  EXPECT_NE(cc::math::ToPrimeField(random.value()), a_);
}

}  // namespace tachyon::cc::math
