#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq2.h"

namespace tachyon::math {

namespace {

class Fp2Test : public testing::Test {
 public:
  static void SetUpTestSuite() { bn254::Fq2::Init(); }
};

}  // namespace

TEST_F(Fp2Test, TypeTest) {
  EXPECT_TRUE((std::is_same_v<bn254::Fq2::BaseField, bn254::Fq>));
  EXPECT_TRUE((std::is_same_v<bn254::Fq2::BasePrimeField, bn254::Fq>));
}

}  // namespace tachyon::math
