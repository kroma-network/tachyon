#include "tachyon/math/elliptic_curves/secp/secp256k1/fr.h"

#include "gtest/gtest.h"

namespace tachyon::math {
namespace secp256k1 {

TEST(Fr, Init) {
  Fr::Config::Init();
  EXPECT_EQ(Fr::Config::kOne, Fr(1).ToMontgomery());
}

}  // namespace secp256k1
}  // namespace tachyon::math
