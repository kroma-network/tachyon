#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"

#include "gtest/gtest.h"

namespace tachyon::math {
namespace bls12_381 {

TEST(Fr, Init) {
  Fr::Config::Init();
  EXPECT_EQ(Fr::Config::kOne, Fr(1).ToMontgomery());
}

}  // namespace bls12_381
}  // namespace tachyon::math
