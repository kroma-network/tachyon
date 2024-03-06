#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/extension_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fp12.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fp2.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fp6.h"
#include "tachyon/cc/math/finite_fields/extension_field_conversions.h"
// #include "tachyon/math/elliptic_curves/bn/bn254/fq2.h"

namespace tachyon {
namespace {

class ExtensionFieldTest : public testing::Test {
 public:
  void SetUp() override {
    native_a2_ = math::bn254::Fq2::Random();
    native_b2_ = math::bn254::Fq2::Random();
    native_a6_ = math::bn254::Fq6::Random();
    native_b6_ = math::bn254::Fq6::Random();
    native_a12_ = math::bn254::Fq12::Random();
    native_b12_ = math::bn254::Fq12::Random();

    c_a2_ = cc::math::c_cast(native_a2_);
    c_b2_ = cc::math::c_cast(native_b2_);
    c_a6_ = cc::math::c_cast(native_a6_);
    c_b6_ = cc::math::c_cast(native_b6_);
    c_a12_ = cc::math::c_cast(native_a12_);
    c_b12_ = cc::math::c_cast(native_b12_);
  }

 protected:
  math::bn254::Fq2 native_a2_;
  math::bn254::Fq2 native_b2_;
  math::bn254::Fq6 native_a6_;
  math::bn254::Fq6 native_b6_;
  math::bn254::Fq12 native_a12_;
  math::bn254::Fq12 native_b12_;

  tachyon_bn254_fp2 c_a2_;
  tachyon_bn254_fp2 c_b2_;
  tachyon_bn254_fp6 c_a6_;
  tachyon_bn254_fp6 c_b6_;
  tachyon_bn254_fp12 c_a12_;
  tachyon_bn254_fp12 c_b12_;
};

}  // namespace

TEST_F(ExtensionFieldTest, Zero) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_zero();
  EXPECT_TRUE(cc::math::native_cast(c_ret2).IsZero());

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_zero();
  EXPECT_TRUE(cc::math::native_cast(c_ret6).IsZero());

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_zero();
  EXPECT_TRUE(cc::math::native_cast(c_ret12).IsZero());
}

TEST_F(ExtensionFieldTest, One) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_one();
  EXPECT_TRUE(cc::math::native_cast(c_ret2).IsOne());

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_one();
  EXPECT_TRUE(cc::math::native_cast(c_ret6).IsOne());

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_one();
  EXPECT_TRUE(cc::math::native_cast(c_ret12).IsOne());
}

TEST_F(ExtensionFieldTest, Random) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_random();
  EXPECT_NE(cc::math::native_cast(c_ret2), native_a2_);

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_random();
  EXPECT_NE(cc::math::native_cast(c_ret6), native_a6_);

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_random();
  EXPECT_NE(cc::math::native_cast(c_ret12), native_a12_);
}

TEST_F(ExtensionFieldTest, Dbl) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_dbl(&c_a2_);
  EXPECT_EQ(cc::math::native_cast(c_ret2), native_a2_.Double());

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_dbl(&c_a6_);
  EXPECT_EQ(cc::math::native_cast(c_ret6), native_a6_.Double());

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_dbl(&c_a12_);
  EXPECT_EQ(cc::math::native_cast(c_ret12), native_a12_.Double());
}

TEST_F(ExtensionFieldTest, Neg) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_neg(&c_a2_);
  EXPECT_EQ(cc::math::native_cast(c_ret2), -native_a2_);

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_neg(&c_a6_);
  EXPECT_EQ(cc::math::native_cast(c_ret6), -native_a6_);

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_neg(&c_a12_);
  EXPECT_EQ(cc::math::native_cast(c_ret12), -native_a12_);
}

TEST_F(ExtensionFieldTest, Sqr) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_sqr(&c_a2_);
  EXPECT_EQ(cc::math::native_cast(c_ret2), native_a2_.Square());

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_sqr(&c_a6_);
  EXPECT_EQ(cc::math::native_cast(c_ret6), native_a6_.Square());

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_sqr(&c_a12_);
  EXPECT_EQ(cc::math::native_cast(c_ret12), native_a12_.Square());
}

TEST_F(ExtensionFieldTest, Inv) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_inv(&c_a2_);
  EXPECT_EQ(cc::math::native_cast(c_ret2), native_a2_.Inverse());

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_inv(&c_a6_);
  EXPECT_EQ(cc::math::native_cast(c_ret6), native_a6_.Inverse());

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_inv(&c_a12_);
  EXPECT_EQ(cc::math::native_cast(c_ret12), native_a12_.Inverse());
}

TEST_F(ExtensionFieldTest, Add) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_add(&c_a2_, &c_b2_);
  EXPECT_EQ(cc::math::native_cast(c_ret2), native_a2_ + native_b2_);

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_add(&c_a6_, &c_b6_);
  EXPECT_EQ(cc::math::native_cast(c_ret6), native_a6_ + native_b6_);

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_add(&c_a12_, &c_b12_);
  EXPECT_EQ(cc::math::native_cast(c_ret12), native_a12_ + native_b12_);
}

TEST_F(ExtensionFieldTest, Sub) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_sub(&c_a2_, &c_b2_);
  EXPECT_EQ(cc::math::native_cast(c_ret2), native_a2_ - native_b2_);

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_sub(&c_a6_, &c_b6_);
  EXPECT_EQ(cc::math::native_cast(c_ret6), native_a6_ - native_b6_);

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_sub(&c_a12_, &c_b12_);
  EXPECT_EQ(cc::math::native_cast(c_ret12), native_a12_ - native_b12_);
}

TEST_F(ExtensionFieldTest, Mul) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_mul(&c_a2_, &c_b2_);
  EXPECT_EQ(cc::math::native_cast(c_ret2), native_a2_ * native_b2_);

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_mul(&c_a6_, &c_b6_);
  EXPECT_EQ(cc::math::native_cast(c_ret6), native_a6_ * native_b6_);

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_mul(&c_a12_, &c_b12_);
  EXPECT_EQ(cc::math::native_cast(c_ret12), native_a12_ * native_b12_);
}

TEST_F(ExtensionFieldTest, Div) {
  tachyon_bn254_fp2 c_ret2 = tachyon_bn254_fp2_div(&c_a2_, &c_b2_);
  EXPECT_EQ(cc::math::native_cast(c_ret2), native_a2_ / native_b2_);

  tachyon_bn254_fp6 c_ret6 = tachyon_bn254_fp6_div(&c_a6_, &c_b6_);
  EXPECT_EQ(cc::math::native_cast(c_ret6), native_a6_ / native_b6_);

  tachyon_bn254_fp12 c_ret12 = tachyon_bn254_fp12_div(&c_a12_, &c_b12_);
  EXPECT_EQ(cc::math::native_cast(c_ret12), native_a12_ / native_b12_);
}

TEST_F(ExtensionFieldTest, Eq) {
  EXPECT_EQ(tachyon_bn254_fp2_eq(&c_a2_, &c_b2_), native_a2_ == native_b2_);
  EXPECT_EQ(tachyon_bn254_fp6_eq(&c_a6_, &c_b6_), native_a6_ == native_b6_);
  EXPECT_EQ(tachyon_bn254_fp12_eq(&c_a12_, &c_b12_),
            native_a12_ == native_b12_);
}

TEST_F(ExtensionFieldTest, Ne) {
  EXPECT_EQ(tachyon_bn254_fp2_ne(&c_a2_, &c_b2_), native_a2_ != native_b2_);
  EXPECT_EQ(tachyon_bn254_fp6_ne(&c_a6_, &c_b6_), native_a6_ != native_b6_);
  EXPECT_EQ(tachyon_bn254_fp12_ne(&c_a12_, &c_b12_),
            native_a12_ != native_b12_);
}

TEST_F(ExtensionFieldTest, Gt) {
  EXPECT_EQ(tachyon_bn254_fp2_gt(&c_a2_, &c_b2_), native_a2_ > native_b2_);
  EXPECT_EQ(tachyon_bn254_fp6_gt(&c_a6_, &c_b6_), native_a6_ > native_b6_);
  EXPECT_EQ(tachyon_bn254_fp12_gt(&c_a12_, &c_b12_), native_a12_ > native_b12_);
}

TEST_F(ExtensionFieldTest, Ge) {
  EXPECT_EQ(tachyon_bn254_fp2_ge(&c_a2_, &c_b2_), native_a2_ >= native_b2_);
  EXPECT_EQ(tachyon_bn254_fp6_ge(&c_a6_, &c_b6_), native_a6_ >= native_b6_);
  EXPECT_EQ(tachyon_bn254_fp12_ge(&c_a12_, &c_b12_),
            native_a12_ >= native_b12_);
}

TEST_F(ExtensionFieldTest, Lt) {
  EXPECT_EQ(tachyon_bn254_fp2_lt(&c_a2_, &c_b2_), native_a2_ < native_b2_);
  EXPECT_EQ(tachyon_bn254_fp6_lt(&c_a6_, &c_b6_), native_a6_ < native_b6_);
  EXPECT_EQ(tachyon_bn254_fp12_lt(&c_a12_, &c_b12_), native_a12_ < native_b12_);
}

TEST_F(ExtensionFieldTest, Le) {
  EXPECT_EQ(tachyon_bn254_fp2_le(&c_a2_, &c_b2_), native_a2_ <= native_b2_);
  EXPECT_EQ(tachyon_bn254_fp6_le(&c_a6_, &c_b6_), native_a6_ <= native_b6_);
  EXPECT_EQ(tachyon_bn254_fp12_le(&c_a12_, &c_b12_),
            native_a12_ <= native_b12_);
}

}  // namespace tachyon
