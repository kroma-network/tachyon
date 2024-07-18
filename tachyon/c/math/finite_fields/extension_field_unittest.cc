#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq12_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq2_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fq6_type_traits.h"

namespace tachyon {
namespace {

class ExtensionFieldTest : public testing::Test {
 public:
  void SetUp() override {
    native_fq2_a_ = math::bn254::Fq2::Random();
    native_fq2_b_ = math::bn254::Fq2::Random();
    native_fq6_a_ = math::bn254::Fq6::Random();
    native_fq6_b_ = math::bn254::Fq6::Random();
    native_fq12_a_ = math::bn254::Fq12::Random();
    native_fq12_b_ = math::bn254::Fq12::Random();

    c_fq2_a = c::base::c_cast(native_fq2_a_);
    c_fq2_b_ = c::base::c_cast(native_fq2_b_);
    c_fq6_a = c::base::c_cast(native_fq6_a_);
    c_fq6_b_ = c::base::c_cast(native_fq6_b_);
    c_fq12_a = c::base::c_cast(native_fq12_a_);
    c_fq12_b2_ = c::base::c_cast(native_fq12_b_);
  }

 protected:
  math::bn254::Fq2 native_fq2_a_;
  math::bn254::Fq2 native_fq2_b_;
  math::bn254::Fq6 native_fq6_a_;
  math::bn254::Fq6 native_fq6_b_;
  math::bn254::Fq12 native_fq12_a_;
  math::bn254::Fq12 native_fq12_b_;

  tachyon_bn254_fq2 c_fq2_a;
  tachyon_bn254_fq2 c_fq2_b_;
  tachyon_bn254_fq6 c_fq6_a;
  tachyon_bn254_fq6 c_fq6_b_;
  tachyon_bn254_fq12 c_fq12_a;
  tachyon_bn254_fq12 c_fq12_b2_;
};

}  // namespace

TEST_F(ExtensionFieldTest, Zero) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret2).IsZero());

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret6).IsZero());

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret12).IsZero());
}

TEST_F(ExtensionFieldTest, One) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_one();
  EXPECT_TRUE(c::base::native_cast(c_ret2).IsOne());

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_one();
  EXPECT_TRUE(c::base::native_cast(c_ret6).IsOne());

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_one();
  EXPECT_TRUE(c::base::native_cast(c_ret12).IsOne());
}

TEST_F(ExtensionFieldTest, MinusOne) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_minus_one();
  EXPECT_TRUE(c::base::native_cast(c_ret2).IsMinusOne());

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_minus_one();
  EXPECT_TRUE(c::base::native_cast(c_ret6).IsMinusOne());

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_minus_one();
  EXPECT_TRUE(c::base::native_cast(c_ret12).IsMinusOne());
}

TEST_F(ExtensionFieldTest, Random) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_random();
  EXPECT_NE(c::base::native_cast(c_ret2), native_fq2_a_);

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_random();
  EXPECT_NE(c::base::native_cast(c_ret6), native_fq6_a_);

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_random();
  EXPECT_NE(c::base::native_cast(c_ret12), native_fq12_a_);
}

TEST_F(ExtensionFieldTest, Dbl) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_dbl(&c_fq2_a);
  EXPECT_EQ(c::base::native_cast(c_ret2), native_fq2_a_.Double());

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_dbl(&c_fq6_a);
  EXPECT_EQ(c::base::native_cast(c_ret6), native_fq6_a_.Double());

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_dbl(&c_fq12_a);
  EXPECT_EQ(c::base::native_cast(c_ret12), native_fq12_a_.Double());
}

TEST_F(ExtensionFieldTest, Neg) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_neg(&c_fq2_a);
  EXPECT_EQ(c::base::native_cast(c_ret2), -native_fq2_a_);

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_neg(&c_fq6_a);
  EXPECT_EQ(c::base::native_cast(c_ret6), -native_fq6_a_);

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_neg(&c_fq12_a);
  EXPECT_EQ(c::base::native_cast(c_ret12), -native_fq12_a_);
}

TEST_F(ExtensionFieldTest, Sqr) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_sqr(&c_fq2_a);
  EXPECT_EQ(c::base::native_cast(c_ret2), native_fq2_a_.Square());

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_sqr(&c_fq6_a);
  EXPECT_EQ(c::base::native_cast(c_ret6), native_fq6_a_.Square());

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_sqr(&c_fq12_a);
  EXPECT_EQ(c::base::native_cast(c_ret12), native_fq12_a_.Square());
}

TEST_F(ExtensionFieldTest, Inv) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_inv(&c_fq2_a);
  EXPECT_EQ(c::base::native_cast(c_ret2), native_fq2_a_.Inverse());

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_inv(&c_fq6_a);
  EXPECT_EQ(c::base::native_cast(c_ret6), native_fq6_a_.Inverse());

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_inv(&c_fq12_a);
  EXPECT_EQ(c::base::native_cast(c_ret12), native_fq12_a_.Inverse());
}

TEST_F(ExtensionFieldTest, Add) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_add(&c_fq2_a, &c_fq2_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), native_fq2_a_ + native_fq2_b_);

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_add(&c_fq6_a, &c_fq6_b_);
  EXPECT_EQ(c::base::native_cast(c_ret6), native_fq6_a_ + native_fq6_b_);

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_add(&c_fq12_a, &c_fq12_b2_);
  EXPECT_EQ(c::base::native_cast(c_ret12), native_fq12_a_ + native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Sub) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_sub(&c_fq2_a, &c_fq2_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), native_fq2_a_ - native_fq2_b_);

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_sub(&c_fq6_a, &c_fq6_b_);
  EXPECT_EQ(c::base::native_cast(c_ret6), native_fq6_a_ - native_fq6_b_);

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_sub(&c_fq12_a, &c_fq12_b2_);
  EXPECT_EQ(c::base::native_cast(c_ret12), native_fq12_a_ - native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Mul) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_mul(&c_fq2_a, &c_fq2_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), native_fq2_a_ * native_fq2_b_);

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_mul(&c_fq6_a, &c_fq6_b_);
  EXPECT_EQ(c::base::native_cast(c_ret6), native_fq6_a_ * native_fq6_b_);

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_mul(&c_fq12_a, &c_fq12_b2_);
  EXPECT_EQ(c::base::native_cast(c_ret12), native_fq12_a_ * native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Div) {
  tachyon_bn254_fq2 c_ret2 = tachyon_bn254_fq2_div(&c_fq2_a, &c_fq2_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), native_fq2_a_ / native_fq2_b_);

  tachyon_bn254_fq6 c_ret6 = tachyon_bn254_fq6_div(&c_fq6_a, &c_fq6_b_);
  EXPECT_EQ(c::base::native_cast(c_ret6), native_fq6_a_ / native_fq6_b_);

  tachyon_bn254_fq12 c_ret12 = tachyon_bn254_fq12_div(&c_fq12_a, &c_fq12_b2_);
  EXPECT_EQ(c::base::native_cast(c_ret12), native_fq12_a_ / native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Eq) {
  EXPECT_EQ(tachyon_bn254_fq2_eq(&c_fq2_a, &c_fq2_b_),
            native_fq2_a_ == native_fq2_b_);
  EXPECT_EQ(tachyon_bn254_fq6_eq(&c_fq6_a, &c_fq6_b_),
            native_fq6_a_ == native_fq6_b_);
  EXPECT_EQ(tachyon_bn254_fq12_eq(&c_fq12_a, &c_fq12_b2_),
            native_fq12_a_ == native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Ne) {
  EXPECT_EQ(tachyon_bn254_fq2_ne(&c_fq2_a, &c_fq2_b_),
            native_fq2_a_ != native_fq2_b_);
  EXPECT_EQ(tachyon_bn254_fq6_ne(&c_fq6_a, &c_fq6_b_),
            native_fq6_a_ != native_fq6_b_);
  EXPECT_EQ(tachyon_bn254_fq12_ne(&c_fq12_a, &c_fq12_b2_),
            native_fq12_a_ != native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Gt) {
  EXPECT_EQ(tachyon_bn254_fq2_gt(&c_fq2_a, &c_fq2_b_),
            native_fq2_a_ > native_fq2_b_);
  EXPECT_EQ(tachyon_bn254_fq6_gt(&c_fq6_a, &c_fq6_b_),
            native_fq6_a_ > native_fq6_b_);
  EXPECT_EQ(tachyon_bn254_fq12_gt(&c_fq12_a, &c_fq12_b2_),
            native_fq12_a_ > native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Ge) {
  EXPECT_EQ(tachyon_bn254_fq2_ge(&c_fq2_a, &c_fq2_b_),
            native_fq2_a_ >= native_fq2_b_);
  EXPECT_EQ(tachyon_bn254_fq6_ge(&c_fq6_a, &c_fq6_b_),
            native_fq6_a_ >= native_fq6_b_);
  EXPECT_EQ(tachyon_bn254_fq12_ge(&c_fq12_a, &c_fq12_b2_),
            native_fq12_a_ >= native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Lt) {
  EXPECT_EQ(tachyon_bn254_fq2_lt(&c_fq2_a, &c_fq2_b_),
            native_fq2_a_ < native_fq2_b_);
  EXPECT_EQ(tachyon_bn254_fq6_lt(&c_fq6_a, &c_fq6_b_),
            native_fq6_a_ < native_fq6_b_);
  EXPECT_EQ(tachyon_bn254_fq12_lt(&c_fq12_a, &c_fq12_b2_),
            native_fq12_a_ < native_fq12_b_);
}

TEST_F(ExtensionFieldTest, Le) {
  EXPECT_EQ(tachyon_bn254_fq2_le(&c_fq2_a, &c_fq2_b_),
            native_fq2_a_ <= native_fq2_b_);
  EXPECT_EQ(tachyon_bn254_fq6_le(&c_fq6_a, &c_fq6_b_),
            native_fq6_a_ <= native_fq6_b_);
  EXPECT_EQ(tachyon_bn254_fq12_le(&c_fq12_a, &c_fq12_b2_),
            native_fq12_a_ <= native_fq12_b_);
}

}  // namespace tachyon
