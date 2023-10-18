#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq12.h"

namespace tachyon::math {

namespace {

class Fp12Test : public testing::Test {
 public:
  static void SetUpTestSuite() { bn254::Fq12::Init(); }
};

}  // namespace

TEST_F(Fp12Test, TypeTest) {
  EXPECT_TRUE((std::is_same_v<bn254::Fq12::BaseField, bn254::Fq6>));
  EXPECT_TRUE((std::is_same_v<bn254::Fq12::BasePrimeField, bn254::Fq>));
}

}  // namespace tachyon::math
