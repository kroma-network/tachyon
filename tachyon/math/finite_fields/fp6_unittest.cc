#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq6.h"

namespace tachyon::math {

namespace {

class Fp6Test : public testing::Test {
 public:
  static void SetUpTestSuite() { bn254::Fq6::Init(); }
};

}  // namespace

TEST_F(Fp6Test, TypeTest) {
  EXPECT_TRUE((std::is_same_v<bn254::Fq6::BaseField, bn254::Fq2>));
  EXPECT_TRUE((std::is_same_v<bn254::Fq6::BasePrimeField, bn254::Fq>));
}

}  // namespace tachyon::math
