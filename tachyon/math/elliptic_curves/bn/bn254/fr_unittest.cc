#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

#include "gtest/gtest.h"

namespace tachyon::math {
namespace bn254 {

TEST(Fr, Init) {
  Fr::Config::Init();
  EXPECT_EQ(Fr::Config::kOne, Fr(1).ToMontgomery());
}

}  // namespace bn254
}  // namespace tachyon::math
