#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_prime_field_traits.h"
#include "tachyon/cc/math/finite_fields/prime_field_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::c::math {

namespace {

class PrimeFieldTest : public testing::Test {
 public:
  void SetUp() override {
    a_ = tachyon::math::bn254::Fr::Random();
    b_ = tachyon::math::bn254::Fr::Random();

    c_a_ = cc::math::ToCPrimeField(a_);
    c_b_ = cc::math::ToCPrimeField(b_);
  }

 protected:
  tachyon::math::bn254::Fr a_;
  tachyon::math::bn254::Fr b_;
  tachyon_bn254_fr c_a_;
  tachyon_bn254_fr c_b_;
};

}  // namespace

TEST_F(PrimeFieldTest, Add) {
  tachyon_bn254_fr c_ret = tachyon_bn254_fr_add(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToPrimeField(c_ret), a_ + b_);
}

TEST_F(PrimeFieldTest, Sub) {
  tachyon_bn254_fr c_ret = tachyon_bn254_fr_sub(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToPrimeField(c_ret), a_ - b_);
}

TEST_F(PrimeFieldTest, Mul) {
  tachyon_bn254_fr c_ret = tachyon_bn254_fr_mul(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToPrimeField(c_ret), a_ * b_);
}

TEST_F(PrimeFieldTest, Div) {
  tachyon_bn254_fr c_ret = tachyon_bn254_fr_div(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToPrimeField(c_ret), a_ / b_);
}

}  // namespace tachyon::c::math
